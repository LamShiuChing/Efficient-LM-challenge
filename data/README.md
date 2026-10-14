# data/

| File | What |
|---|---|
| `prepare_fineweb.py` | FineWeb `sample-10BT` -> GPT-2 token shards in `data_cache/fineweb10B/` + `manifest.json` |
| `loader.py` | shard format (`shard_bytes`, `read_shard`) and `ShardLoader(pattern, B, T)`; `python data/loader.py --selftest` |
| `rename_entities.py` | Phase 1a knowledge-light arms: spaCy NER over the same docs, then entity-renamed shards in `data_cache/renamed/<arm>/` (`r50`, `r80`, `r100` surrogates for the rarest entities by doc frequency; `tag100` typed tags); `--selftest`. Docstring has the details |

## Source

- HF dataset `HuggingFaceFW/fineweb`, config `sample-10BT` (files `sample/10BT/*.parquet`, 15 files, ~30.6 GB),
  the same sample llm.c uses (`dev/data/fineweb.py`, `-t classic -v 10B`). The dataset commit sha is resolved at
  first run (or pinned with `--revision`) and recorded in the manifest along with each parquet's LFS sha256.
- Raw parquet stays in the HF cache (`data_cache/hf`, set as `HF_HOME` by the script). Later phases need the raw text.

## Tokenization (matches llm.c)

- tiktoken `gpt2`, `encode_ordinary` (special-token strings in the text are encoded as plain text).
- `<|endoftext|>` (50256) is **prepended** to every document.
- Documents are concatenated in dataset order (parquet files sorted by name, rows in order; same order as
  `load_dataset(..., split="train")`). Shards are exactly `shard_size` (1e8) tokens, documents are split across
  shard boundaries, the last shard holds the remainder. A document that exactly fills a shard closes it (llm.c's `<`).
- Shard 0 is validation: `fineweb_val_000000.bin`, then `fineweb_train_000001.bin`, `..._000002.bin`, ...

## Shard format (llm.c `data_common.write_datafile`, gpt-2)

```
bytes 0..1023 : 256 x int32 little-endian header
                [0] magic   = 20240520
                [1] version = 1
                [2] num_tokens
                [3..255] = 0
bytes 1024..  : num_tokens x uint16 token ids
```
File size = 1024 + 2 * num_tokens.

## Manifest

`manifest.json` holds: dataset id, config, revision (commit sha), source parquet list with sha256, tokenizer,
format, settings (`shard_size`, `max_tokens`, `limit_docs`), and per shard: file, num_tokens, sha256, the resume
pointer `end = [parquet index, row, tokens of that doc already written]`, and `docs_started`. It has no timestamps,
so its sha256 is a stable **data manifest hash** for experiment records (protocol §8):
`python -c "import hashlib;print(hashlib.sha256(open('data_cache/fineweb10B/manifest.json','rb').read()).hexdigest())"`.

## Running

```
python data/prepare_fineweb.py                       # full run -> data_cache/fineweb10B/
python data/prepare_fineweb.py --out_dir data_cache/fineweb_test --max_tokens 20000000 --shard_size 5000000
```
Re-running the same command resumes after the last shard in the manifest (settings must match; changing them
requires a new `--out_dir`). Shard writes are atomic (`.tmp` + rename), so a killed run loses at most the shard in
progress. Measured on this machine (32 cores): ~20M tok/s tokenizing, ~40 MB/s unauthenticated download.

## Renamed arms (Phase 1a)

`bash scripts/vast_job.sh rename` (CPU-only job) runs `rename_entities.py ner`, then `rename --arm` for each arm, and
uploads to HF `lamshiu/efficient-lm-fineweb-renamed` (`ner/`, `<arm>/`). Same docs, order and tokenization as above;
the val shard holds exactly the first 143,468 docs (whole), train shards start at the next doc. Each arm's
`manifest.json` records the df threshold, achieved mention fraction, `token_ratio` vs the original text,
`frac_tokens_changed`, `propagated` and `residual_per_renamed`.
Train on an arm: `DATA_ARM=r50 bash scripts/vast_job.sh train <config> '--train_pattern=data_cache/r50/fineweb_train_*.bin' '--val_pattern=data_cache/r50/fineweb_val_*.bin'`.

Limitations (measured on the first 240 docs):
- **NER false positives** (`en_core_web_sm`) rename ordinary words, e.g. "Java" (PERSON, the language), "Mathematica"
  (GPE), "Working Together" (ORG, a headline), "Goodwill" (PERSON), "Oakland" (ORG), "Mobile", "Group", "Today",
  "thou", "Clicks", "Vessel". Propagation (below) spreads a false positive to every capitalized repeat in its doc.
- **NER misses repeats**: ~15% of renamed surface strings still occurred elsewhere in the same doc. Propagation
  (replace every whole-word, case-sensitive repeat of a renamed surface that starts uppercase, >= 2 letters) cuts
  this to ~2% (`residual_per_renamed`); what is left is lowercase variants, 1-letter names, and repeats inside kept spans.
- **Token count**: surrogates are matched to the original word's GPT-2 token count, but never use a common word
  (`ner/words_df.parquet`, top 100k) or any word of any NER entity. If that blocks most 1-token syllables
  (`free_one_token_syllables` in the arm manifest), 1-token names get 2-token surrogates: `token_ratio` was 1.01
  with the 240-doc blocklist and up to ~1.06 (r100) if every 1-token syllable is blocked. `tag100` is ~1.055
  (`PERSON_1` is 3 tokens).
- Entity spans are ~10% of all tokens, so r100 changes ~12% of tokens (with propagation), tag100 ~16%.
