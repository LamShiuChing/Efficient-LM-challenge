# Eval suite: `evals-v0.2-dev`

This is the frozen spec of the eval harness. Code: `evals/harness.py` (the version string is `SUITE_VERSION`). Any change to a task, prompt, split or scoring bumps the version, and every baseline is then re-scored (protocol §3). `-dev` means the suite is not frozen yet. It becomes `evals-v1` at the end of Phase 0.

```
python -m evals.run --model hf:gpt2|ckpt:<path> --tasks popqa_closed,popqa_oracle,synth_1hop,synth_2hop [--split dev|test] [--limit N] --out res.json
python -m evals.test_harness   # offline check of the scoring math and item builders, no network
```

`HF_HOME` defaults to `data_cache/hf`.

**Local CPU runs need `CUDA_VISIBLE_DEVICES=-1`** on this Windows build (torch 2.11 + cu128). An empty `CUDA_VISIBLE_DEVICES=""` does not hide the GPU there: `torch.cuda.is_available()` stays True and the model runs on `cuda:0`.

## Changelog

- `evals-v0-dev`: hellaswag, piqa, winogrande, arc_easy, lambada, blimp.
- `evals-v0.2-dev`: adds popqa_closed, popqa_oracle, synth_1hop, synth_2hop (Phase 1a), per-bucket metrics and `majority_baseline` for popqa, and acc_norm for the new tasks. Scores on the six old tasks are unchanged.

## Scoring (all tasks)

Every item is a list of choices `(context, continuation)` plus a gold index. Each choice is scored by the summed log-probability of the continuation given the context.

- **Tokenization.** tiktoken `gpt2` `encode_ordinary`. Context and continuation are tokenized separately, as in llm.c, and the continuation carries its leading space. An empty context is replaced by `<|endoftext|>`. Rows are left-truncated to the 1024-token window.
- **acc**: the argmax of the summed log-prob is the gold choice (ties go to the first choice).
- **acc_norm**: the argmax of log-prob per continuation token. It is reported for tasks whose choices differ in length: hellaswag, piqa, arc_easy, popqa_*, synth_*. Note that this is per *token*, not lm-eval's per *byte*.
- **gold_logprob_per_tok**: the mean over items of the gold continuation's log-prob divided by its token count. It is the continuous metric to rank models by at small scale (protocol §5).
- **lambada**: acc means the whole target word is the greedy continuation. ppl is `exp(-mean target-word log-prob)`, as in lm-eval.
- **CIs**: 95% percentile bootstrap over items (1000 resamples, seed 0) for every metric.
- **popqa buckets**: `buckets.{tail,torso,head}` repeats every metric on the items in each subject-popularity bucket.
- **popqa majority_baseline**: the accuracy, on the same items, of always picking the choice that is most often gold for the item's relation (counted over all 13065 deduped questions). It is reported overall and per bucket. This is the floor to compare against, not 25%.

## Splits: dev vs test

Use **dev** for every decision. **test** is touched only for the final numbers of a preregistered comparison (protocol §4). Where a dataset has hidden test labels, its official validation set serves as our test split, and dev is carved from train with a seeded subset (`SEED = 0`).

## Tasks

| Task | HF repo @ revision | dev | test | Prompt (context → continuation) | Metrics |
|---|---|---|---|---|---|
| hellaswag | `Rowan/hellaswag` @ `218ec52e` | 10042 seeded rows of train | validation | `{ctx}` → ` {ending}` (4) | acc, acc_norm |
| piqa | `baber/piqa` @ `142f6d73` | 5000 seeded rows of train | validation | `Question: {goal}\nAnswer:` → ` {sol}` (2) | acc, acc_norm |
| winogrande | `allenai/winogrande` (winogrande_xl) @ `01e74176` | 5000 seeded rows of train | validation | partial scoring: sentence up to `_` with the option filled in → ` {rest}` (2) | acc |
| arc_easy | `allenai/ai2_arc` (ARC-Easy) @ `210d026f` | train + validation | test | `Question: {q}\nAnswer:` → ` {choice}` | acc, acc_norm |
| lambada | test `EleutherAI/lambada_openai` (default) @ `900124bf`; dev `cimec/lambada` @ `5953bd97` | cimec validation (lowercased, PTB-tokenized: **not on the test scale**) | lambada_openai test | `{text minus last word}` → ` {last word}` | acc (greedy), ppl |
| blimp | `nyu-mll/blimp` (67 configs) @ `877fba08` | seeded 500 of 1000 per paradigm | the other 500 | `<eot>` → `{good}` vs `{bad}` | acc |
| popqa_closed | `akariasai/PopQA` @ `098765c7` | seeded 50% of deduped questions (6532) | the other 50% (6533) | `Q: {question}\nA:` → ` {answer}` (4) | acc, acc_norm, per bucket, majority_baseline |
| popqa_oracle | same items and choices as popqa_closed | same | same | `{4 facts, one per choice}\nQ: {question}\nA:` → ` {answer}` (4) | acc, acc_norm, per bucket, majority_baseline |
| synth_1hop | generated (no download) | 1000 items, seeds `synth-dev-{i}` | 1000 items, seeds `synth-test-{i}` | `{8 facts} {P} works in` → ` {city}` (4) | acc, acc_norm |
| synth_2hop | generated | same contexts as synth_1hop | same | `{8 facts} {P} works in a city in the country of` → ` {country}` (4) | acc, acc_norm |

All revisions are pinned in `REV` in `harness.py` as full commit hashes. They were verified on the Hub on 2026-10-13, and PopQA on 2026-10-14.

### PopQA (Mallen et al. 2023)

- **Source.** https://huggingface.co/datasets/akariasai/PopQA has a single `test.tsv` (14267 rows, 16 relations). Its fields are `id, subj, prop, obj, subj_id, prop_id, obj_id, s_aliases, o_aliases, s_uri, o_uri, s_wiki_title, o_wiki_title, s_pop, o_pop, question, possible_answers`, where `possible_answers` is a JSON list.
- **Items.**
  - Rows with an empty `subj` are dropped (3 rows).
  - Repeated questions (same subj and prop, 1199 rows) are deduped: the first row wins and the alias sets are unioned. This leaves 13065 questions.
  - The gold answer is the first entry of `possible_answers`.
- **Distractors.** 3 objects are taken from other questions with the same `prop`. They are sampled by row, so an object is drawn as often as it is gold. A distractor is never an alias of the gold (case-insensitive) and never repeats. The gold position is random. Everything is seeded by `(SEED, row id)`, so items do not depend on the split or on `--limit`. No question is dropped for lack of distractors.
- **Split.** A seeded 50/50 permutation of the 13065 questions. The order is kept permuted, so `--limit N` gives a random sample.
- **Popularity buckets.** These are `s_pop` terciles over all 14267 raw rows, fixed in code as `POPQA_POP_BOUNDS = (375, 2879)`: tail `<= 375`, torso `<= 2879`, head `> 2879`. Dev has 2296 / 2159 / 2077 items in these buckets.
- **Oracle context.** It has 4 fact sentences, one per choice, all true, in seeded shuffled order, followed by the closed-book prompt. Each uses the relation's template (`POPQA_TEMPLATES`, for example `"{subj} was born in {obj}."`).
  - The gold fact binds the question's subject to the gold answer.
  - Each distractor fact binds another PopQA subject to that distractor. The subject comes from the row the distractor was drawn from, whose gold is that object.
  - Each of the 4 choices is the object of exactly one fact, so copying any entity from the context does not help. The model has to bind subject to object.
- **Chance and priors.** Gold objects are skewed (for example, `country` is mostly the USA), so the floor is **about 28%, not 25%**. The `majority_baseline` field (most-common-gold-for-this-relation) scores 28.3% on all of dev. Compare acc against `majority_baseline` on the same items, not against 25%.

### Synthetic in-context reasoning (synth_1hop, synth_2hop)

- **Entities.** All names are fictional and pronounceable, from a seeded syllable generator (`_pseudo`). Countries end in `-ia`. Within an item, no name is a substring of another. The generator is independent of any training-data renaming code.
- **Facts.** Each item has 4 chains, person → city → country, written as 8 facts in shuffled order. The person → city relation is one of `works in`, `lives in` or `was born in`, the same for the whole item. The city → country fact is `{C} is a city in {K}.`
- **Cloze.** The facts are followed by a cloze about one of the 4 people. The phrasing is fixed per relation.
  - 1-hop: `{P} {rel}`, for example `Dantrin lives in`, completed by the city. The 4 choices are the 4 cities, and each appears twice in the context.
  - 2-hop: `{P} {rel} a city in the country of`, completed by the country. The 4 choices are the 4 countries, and each appears once.
  - No sentence contains both the person and the gold country.
  - Because every candidate appears equally often, frequency gives no hint.
- **Pairing.** Item `i` has the same context and person in both tasks, so 2-hop minus 1-hop is a paired difference.
- **What it measures.** There is nothing to memorize. The task tests in-context reasoning only, and it has no closed-book or retrieved condition.

## Closed-book / oracle-context / retrieved mapping (protocol §6)

| Fact eval | closed-book | oracle-context | retrieved |
|---|---|---|---|
| PopQA | `popqa_closed` | `popqa_oracle` (gold fact plus one true fact per distractor) | **not implemented yet**: planned BM25 top-k over the fixed Wikipedia snapshot, before Phase 1c |
| synth_1hop / synth_2hop | n/a (nothing to know) | the task itself | n/a |

Expected pattern for knowledge-light arms: popqa_closed drops, most on tail facts. popqa_oracle and synth_* do not drop.

## Known limitations

- **Near-synonym distractors in PopQA.** Different Wikidata objects can mean nearly the same thing (Catholicism vs Catholic Church, association football vs American football). A rough word-overlap check flags about 3% of items, mostly in sport, capital-of and genre. This is not fixed.
- **The 2-hop cloze phrasing is fixed.** `a city in the country of` reuses the words of the city facts (`is a city in`). That is intended, since it names the chaining step, but it is a single phrasing.
- **No decontamination yet** of the training data against these eval sets (protocol §4). For PopQA, overlap with FineWeb is the point of the manipulation check, so n-gram decontamination does not apply in the usual way.
- **acc_norm is per token.** It favors answers that split into many cheap tokens, such as long multi-word objects and pseudo-names.
