"""Phase 1a knowledge-light data: rename named entities in FineWeb sample-10BT, tokenized exactly like prepare_fineweb.

Same documents in the same order as data/prepare_fineweb.py (same dataset revision). Two stages, outputs under
--out_dir (default data_cache/renamed/):

  ner              spaCy en_core_web_sm (NER pipe only) over the val docs (the first VAL_DOCS docs = the val shard of
                   lamshiu/efficient-lm-fineweb10B-gpt2) plus following docs until --train_tokens of train text.
                   -> ner/spans_NNNNN.parquet (doc, start, end, label id), ner/df_NNNNN.parquet (per chunk),
                      ner/entity_df.parquet (surface, label, df, mentions), ner/words_df.parquet (top lowercase
                      words by doc frequency: surrogate blocklist), ner/manifest.json.
                   Chunked: re-running the same command resumes after the last finished chunk.
  rename --arm A   r50 / r80 / r100: the rarest (surface, label) keys, by document frequency, up to one df threshold
                   that covers ~50 / 80 / 100% of mentions, get pseudo-word surrogates (consistent within a doc,
                   independent across docs). tag100: every mention becomes a typed tag (PERSON_1, ORG_2, ...).
                   swap50: the r50 keys get *real* names instead: words of other rare entities of the same label
                   (docs/PLAN_1B.md), mapped word by word within a doc so partial mentions stay consistent.
                   -> <arm>/fineweb_val_000000.bin (the renamed val docs, whole docs), fineweb_train_000001.bin, ...
                      (1e8-token llm.c shards, docs split across boundaries), <arm>/manifest.json.

  python data/rename_entities.py ner               # workers: CPU quota/affinity, capped by RAM / 0.6 GB
  python data/rename_entities.py rename --arm r50
  python data/rename_entities.py --selftest
Tiny local run: add --stream (reads only the needed parquet row groups) and small --val_docs/--train_tokens/...
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # the process pool is the parallelism; no BLAS threads per worker

from prepare_fineweb import CONFIG, REPO, ROOT, SUBDIR, tokenize  # first: sets HF_HOME before huggingface imports

import argparse
import bisect
import hashlib
import json
import multiprocessing as mp
import random
import re
import string
import sys
import time
from collections import Counter
from importlib.metadata import version
from itertools import islice

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from loader import shard_bytes

REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"  # revision in lamshiu/efficient-lm-fineweb10B-gpt2 manifest.json
VAL_DOCS = 143_468  # docs_started of that repo's val shard (its last doc continues into train shard 1)
NER_MODEL = "en_core_web_sm"
NON_NER = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "senter"]  # ner has its own tok2vec
LABELS = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]
LABEL_ID = {l: i for i, l in enumerate(LABELS)}
ARMS = {"r50": 0.5, "r80": 0.8, "r100": 1.0, "tag100": 1.0, "swap50": 0.5}  # target fraction of mentions renamed
NAME = re.compile(r"[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}")  # swap pool: plain capitalized names of 1-4 words
PIECE_CHARS = 20_000  # NER runs on pieces cut at newlines: bounds spaCy memory/max_length on huge docs
TOP_WORDS = 100_000  # surrogate words never equal one of the most common lowercase words (ner: words_df.parquet)

WORD = re.compile(r"[^\W\d_]+")  # letter runs: the part of a span that surrogates replace (GPT-2 splits these too)
LEAD = re.compile(r"\s*(?:[Tt]he\s+)?")
TAIL = re.compile(r"(?:['’]s)?\s*$")
ONSETS = "b c d f g h j k l m n p r s t v w z b d k l m n r s t br dr fr gr kr pr tr bl fl gl kl pl sl st sh ch th".split()
VOWELS = "a e i o u a e i o a e ai ea ou io".split()
CODAS = ["", "", "", "", "n", "r", "l", "s", "m", "k", "t", "nd", "rn", "st"]

_enc = None


def ntok(s):
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("gpt2")
    return len(_enc.encode_ordinary(s))


def trim(text, s, e):
    """Drop a leading 'the ' and a trailing possessive 's from an entity span, so both stay in the text."""
    s = LEAD.match(text, s, e).end()
    return s, TAIL.search(text, s, e).start()


def pieces(text, n=PIECE_CHARS):
    """Yield (offset, piece) with pieces <= n chars, cut before a newline (else a space) where possible."""
    off = 0
    while len(text) - off > n:
        cut = text.rfind("\n", off + 1, off + n)
        if cut < 0:
            cut = text.rfind(" ", off + 1, off + n)
        if cut < 0:
            cut = off + n
        yield off, text[off:cut]
        off = cut
    yield off, text[off:]


def source_files(revision):
    info = HfApi().dataset_info(REPO, revision=revision, files_metadata=True)
    files = sorted((s.rfilename, s.lfs.sha256, s.size) for s in info.siblings if s.rfilename.startswith(SUBDIR))
    return info.sha, [{"path": f, "sha256": h, "size": n} for f, h, n in files]


def iter_texts(revision, files, start=0, stream=False):
    """Texts of docs start, start+1, ... in prepare_fineweb's order (parquet files sorted by name, rows in order).
    stream=True reads only the needed row groups over HTTP (tiny local tests) instead of downloading 2 GB files."""
    fs = HfFileSystem() if stream else None
    i = 0
    for f in files:
        src = fs.open(f"datasets/{REPO}@{revision}/{f}") if stream else \
            hf_hub_download(REPO, f, repo_type="dataset", revision=revision)
        pfile = pq.ParquetFile(src)
        if i + pfile.metadata.num_rows <= start:
            i += pfile.metadata.num_rows
            continue
        for rb in pfile.iter_batches(batch_size=1024, columns=["text"]):  # ~1 parquet row group
            texts = rb.column(0).to_pylist()
            k, i = max(0, start - i), i + len(texts)
            yield from texts[k:]


def pseudo_word(rng, word, *avoid):
    """Random pronounceable word shaped like `word`: same case pattern and, where a draw allows, the same GPT-2 token
    count (with a leading space), so renaming keeps the BPE length distribution. Its lowercase is in none of `avoid`
    (real words / entity words). Every 25 draws allow one more syllable, so a blocked short pool falls back to longer
    words (e.g. 2-token surrogates for 1-token names) instead of emitting a blocked word."""
    target, best = ntok(" " + word), None
    for k in range(10_000):
        if word.isupper() and len(word) <= 3 and k < 50:  # initials, short acronyms: random capitals
            s = "".join(rng.choice(string.ascii_uppercase) for _ in word)
        else:  # ~1.4 tokens per syllable for this generator
            n = min(4, max(1, round(0.7 * target))) + k // 25
            s = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(n))
            s = s.upper() if word.isupper() else s if word.islower() else s.capitalize()
        if s.lower() == word.lower() or any(s.lower() in a for a in avoid):
            continue
        cost = abs(ntok(" " + s) - target)
        if best is None or cost < best[0]:
            best = (cost, s)
        if cost == 0 or k >= 50:
            break
    return best[1]


def swap_words(rng, words, label, pool, wmap, used):
    """Real replacement words for the letter runs `words` of one surface: a word already mapped in this doc keeps
    its replacement (so "Lovelace" alone matches "Ada Lovelace"); the rest come position by position from one pool
    name of the same label and word count (else any of the label), never the original word or a replacement already
    used for another word in this doc. Case follows the source word (IBM -> ACME, de -> de-like lowercase)."""
    cands = pool.get((label, len(words))) or pool[label]
    for k in range(50):
        name = cands[rng.randrange(len(cands))]
        new = [wmap.get(w) or name[min(i, len(name) - 1)] for i, w in enumerate(words)]
        fresh = [n for w, n in zip(words, new) if w not in wmap]
        if all(n.lower() != w.lower() for w, n in zip(words, new)) and not (set(fresh) & used) \
                and len(set(fresh)) == len(fresh):
            break
    for w, n in zip(words, new):
        if w not in wmap:
            wmap[w] = n
            used.add(n)
    return [n.upper() if w.isupper() and len(w) > 1 else n.lower() if w.islower() else n for w, n in zip(words, new)]


def rename_doc(text, spans, arm, doc_idx, keep=frozenset(), block=frozenset(), pool=None):
    """Replace entity spans [(start, end, label)] (sorted, disjoint) unless (surface, label) is in `keep`, then also
    every other whole-word, case-sensitive occurrence (outside any span) of a renamed surface that starts with an
    uppercase letter and has >= 2 letters (catches NER misses of repeats), with the same replacement.
    Surrogate arms: each letter run becomes a pseudo-word (digits, punctuation, whitespace stay); the same surface
    string gets the same surrogate within the doc; RNG seeded by (arm, doc_idx). tag100: LABEL_k, k numbered per doc
    per label in order of first appearance; a propagated occurrence takes the surface's first tag.
    Returns (new_text, reps, changed_tokens, residual, propagated): reps[i] = new text of span i or None if kept;
    changed_tokens = GPT-2 tokens of all replacement text; residual = occurrences of a replaced surface string left
    in new_text; propagated = extra occurrences replaced."""
    rng, tags, swap = random.Random(f"{arm}:{doc_idx}"), arm == "tag100", arm.startswith("swap")
    seen, used, count, first, wmap = {}, set(), {}, {}, {}  # first: surface -> its first replacement function
    edits, reps, pos = [], [], 0
    for s, e, lab in spans:
        assert pos <= s < e <= len(text), "spans must be sorted and disjoint"
        pos = e
        surface = " ".join(text[s:e].split())
        if (surface, lab) in keep:
            reps.append(None)
            continue
        if tags:
            if (surface, lab) not in seen:
                count[lab] = count.get(lab, 0) + 1
                seen[surface, lab] = f"{lab}_{count[lab]}"
            first.setdefault(surface, lambda t, tag=seen[surface, lab]: tag)
        elif surface not in seen and swap:
            seen[surface] = swap_words(rng, WORD.findall(surface), lab, pool, wmap, used)
            first[surface] = lambda t, ws=seen[surface]: WORD.sub(lambda m, it=iter(ws): next(it), t)
        elif surface not in seen:
            seen[surface] = [pseudo_word(rng, w, block, used) for w in WORD.findall(surface)]
            used.update(w.lower() for w in seen[surface])
            first[surface] = lambda t, ws=seen[surface]: WORD.sub(lambda m, it=iter(ws): next(it), t)
        new = seen[surface, lab] if tags else first[surface](text[s:e])
        edits.append((s, e, new))
        reps.append(new)
    prop = [x for x in first if x[0].isupper() and sum(map(len, WORD.findall(x))) >= 2]
    n_prop = 0
    if prop:
        starts, ends = [s for s, _, _ in spans], [e for _, e, _ in spans]
        pat = r"(?<!\w)(?:%s)(?!\w)" % "|".join(map(re.escape, sorted(prop, key=len, reverse=True)))
        for m in re.finditer(pat, text):
            i = bisect.bisect_left(ends, m.start() + 1)  # first span ending after m.start()
            if i < len(spans) and starts[i] < m.end():
                continue  # overlaps a span (renamed or kept)
            edits.append((m.start(), m.end(), first[m.group()](m.group())))
            n_prop += 1
        edits.sort()
    out, pos, changed = [], 0, 0
    for s, e, new in edits:
        out += [text[pos:s], new]
        pos = e
        pre = text[s - 1] if s else ""
        if tags:
            changed += ntok((" " if pre == " " else "") + new)
        else:
            for m in WORD.finditer(new):
                c = new[m.start() - 1] if m.start() else pre
                changed += ntok((" " if c == " " else "") + m.group())
    out.append(text[pos:])
    new_text = "".join(out)
    gone = list(first)
    residual = len(re.findall(r"(?<!\w)(?:%s)(?!\w)" % "|".join(map(re.escape, sorted(gone, key=len, reverse=True))),
                              new_text)) if gone else 0
    return new_text, reps, changed, residual, n_prop


def pick_threshold(df, mentions, frac):
    """Rename keys with df <= t, rarest first. Returns (t, achieved mention fraction) with achieved closest to frac."""
    i = np.argsort(df, kind="stable")
    d, c = df[i], np.cumsum(mentions[i])
    last = np.r_[np.nonzero(np.diff(d))[0], len(d) - 1]  # last index of each distinct df
    fr = c[last] / c[-1]
    j = int(np.argmin(np.abs(fr - frac)))
    return int(d[last[j]]), float(fr[j])


# ---------------------------------------------------------------- workers (module-level state, one per process)
_nlp = _arm = _keep = _block = _pool = None


def _init_ner():
    global _nlp
    import spacy
    _nlp = spacy.load(NER_MODEL, exclude=NON_NER)
    assert _nlp.pipe_names == ["ner"], _nlp.pipe_names


def _ner_batch(texts):
    """-> (per doc ([(start, end, label_id, normalized surface)], n_tokens incl. <|endoftext|>),
           Counter of lowercase word -> number of these docs containing it)"""
    words = Counter()
    for t in texts:
        words.update(set(WORD.findall(t.lower())))
    spans = [[] for _ in texts]
    ps = [(p, (i, off)) for i, t in enumerate(texts) for off, p in pieces(t)]
    for doc, (i, off) in _nlp.pipe(ps, as_tuples=True, batch_size=64):
        for ent in doc.ents:
            if ent.label_ in LABEL_ID:
                s, e = trim(doc.text, ent.start_char, ent.end_char)
                surface = " ".join(doc.text[s:e].split())
                if surface:
                    spans[i].append((off + s, off + e, LABEL_ID[ent.label_], surface))
    return [(sp, len(t)) for sp, t in zip(spans, tokenize(texts))], words


def _init_rename(arm, keep, block, pool):
    global _arm, _keep, _block, _pool
    _arm, _keep, _block, _pool = arm, keep, block, pool


def _rename_batch(items):
    out = []
    for d, text, spans in items:
        new, reps, changed, resid, nprop = rename_doc(text, [(s, e, LABELS[l]) for s, e, l in spans], _arm, d,
                                                      _keep, _block, _pool)
        out.append((d, tokenize([new])[0], len(reps), sum(r is not None for r in reps), changed, resid, nprop))
    return out


# ---------------------------------------------------------------- stages
def save_json(obj, path):  # atomic, so a kill never leaves a half-written manifest
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def write_parquet(table, path):
    pq.write_table(table, path + ".tmp")
    os.replace(path + ".tmp", path)



def ner_stage(args):
    d = os.path.join(args.out_dir, "ner")
    os.makedirs(d, exist_ok=True)
    mpath = os.path.join(d, "manifest.json")
    settings = {"val_docs": args.val_docs, "train_tokens": args.train_tokens, "chunk_docs": args.chunk_docs,
                "piece_chars": PIECE_CHARS}
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        assert man["settings"] == settings, f"settings differ from {mpath}: {man['settings']} vs {settings}"
        if man["complete"]:
            print(f"{mpath} says complete; nothing to do")
            return
        print(f"resuming after {len(man['chunks'])} chunks")
    else:
        rev, files = source_files(args.revision)
        man = {"dataset": REPO, "config": CONFIG, "revision": rev, "source_files": files,
               "ner": {"model": NER_MODEL, "model_version": version(NER_MODEL), "spacy": version("spacy"),
                       "pipes": ["ner"], "labels": LABELS,
                       "spans": "entity span minus a leading 'the '/'The ' and a trailing 's; empty spans dropped",
                       "key": "(surface, label), surface = span text whitespace-collapsed and stripped, case kept"},
               "settings": settings, "chunks": [], "complete": False}
    doc0 = man["chunks"][-1]["doc_end"] if man["chunks"] else 0
    train_tok = sum(c["train_tokens"] for c in man["chunks"])
    docs = iter_texts(man["revision"], [f["path"] for f in man["source_files"]], doc0, args.stream)
    t0, docs0 = time.time(), doc0
    with mp.Pool(args.nprocs, initializer=_init_ner) as pool:
        while doc0 < args.val_docs or train_tok < args.train_tokens:
            texts = list(islice(docs, args.chunk_docs))
            assert texts, "ran out of documents"
            # ponytail: one Pool.map per chunk (bounded memory); workers idle briefly at each chunk's tail
            out = pool.map(_ner_batch, [texts[i:i + 32] for i in range(0, len(texts), 32)])
            res = [r for b, _ in out for r in b]
            rows = [(doc0 + i, s, e, l, surf) for i, (sp, _) in enumerate(res) for s, e, l, surf in sp]
            doc, st, en, lab, surf = (list(c) for c in zip(*rows)) if rows else ([],) * 5
            k = len(man["chunks"])
            write_parquet(pa.table({"doc": pa.array(doc, pa.int64()), "start": pa.array(st, pa.int32()),
                                    "end": pa.array(en, pa.int32()), "label": pa.array(lab, pa.uint8())}),
                          os.path.join(d, f"spans_{k:05d}.parquet"))
            ments = pa.table({"surface": pa.array(surf, pa.string()),
                              "label": pa.array([LABELS[l] for l in lab], pa.string()), "doc": pa.array(doc, pa.int64())})
            dfc = ments.group_by(["surface", "label"]).aggregate([("doc", "count_distinct"), ("doc", "count")])
            write_parquet(dfc.select(["surface", "label", "doc_count_distinct", "doc_count"])
                          .rename_columns(["surface", "label", "df", "mentions"]), os.path.join(d, f"df_{k:05d}.parquet"))
            wk, wv = [], []
            for _, w in out:
                wk.extend(w.keys())
                wv.extend(w.values())
            wt = pa.table({"word": pa.array(wk, pa.string()), "df": pa.array(wv, pa.int64())})                 .group_by("word").aggregate([("df", "sum")]).select(["word", "df_sum"]).rename_columns(["word", "df"])
            write_parquet(wt.filter(pc.greater_equal(wt["df"], 2)), os.path.join(d, f"words_{k:05d}.parquet"))
            ntoks = [n for _, n in res]
            nval = max(0, min(len(texts), args.val_docs - doc0))
            man["chunks"].append({"spans": f"spans_{k:05d}.parquet", "df": f"df_{k:05d}.parquet",
                                  "words": f"words_{k:05d}.parquet", "doc_start": doc0,
                                  "doc_end": doc0 + len(texts), "mentions": len(rows),
                                  "val_tokens": sum(ntoks[:nval]), "train_tokens": sum(ntoks[nval:])})
            save_json(man, mpath)
            doc0, train_tok = doc0 + len(texts), train_tok + sum(ntoks[nval:])
            dt = time.time() - t0
            print(f"chunk {k}: docs {doc0:,} | train tokens {train_tok:,} | {(doc0 - docs0) / dt:,.0f} docs/s "
                  f"({(doc0 - docs0) / dt / args.nprocs:,.1f} per process) | {dt / 60:.1f} min", flush=True)
    tab = pa.concat_tables([pq.read_table(os.path.join(d, c["df"])) for c in man["chunks"]])
    tab = tab.group_by(["surface", "label"]).aggregate([("df", "sum"), ("mentions", "sum")]) \
        .select(["surface", "label", "df_sum", "mentions_sum"]).rename_columns(["surface", "label", "df", "mentions"]) \
        .sort_by([("df", "descending"), ("mentions", "descending"), ("surface", "ascending"), ("label", "ascending")])
    write_parquet(tab, os.path.join(d, "entity_df.parquet"))
    by = tab.group_by("label").aggregate([("mentions", "sum"), ("df", "count")]).to_pylist()
    m = tab["mentions"].to_numpy()
    man["stats"] = {"docs": doc0, "val_tokens": sum(c["val_tokens"] for c in man["chunks"]), "train_tokens": train_tok,
                    "mentions": int(m.sum()), "keys": tab.num_rows,
                    "keys_df1": int(pc.sum(pc.equal(tab["df"], 1)).as_py()),
                    "by_label": {r["label"]: {"mentions": r["mentions_sum"], "keys": r["df_count"]}
                                 for r in sorted(by, key=lambda r: LABEL_ID[r["label"]])}}
    man["entity_df"] = {"file": "entity_df.parquet", "sha256": hashlib.sha256(open(os.path.join(d, "entity_df.parquet"), "rb").read()).hexdigest()}
    wt = pa.concat_tables([pq.read_table(os.path.join(d, c["words"])) for c in man["chunks"]])
    wt = wt.group_by("word").aggregate([("df", "sum")]).select(["word", "df_sum"]).rename_columns(["word", "df"])         .sort_by([("df", "descending"), ("word", "ascending")]).slice(0, TOP_WORDS)
    write_parquet(wt, os.path.join(d, "words_df.parquet"))
    man["words_df"] = {"file": "words_df.parquet", "rows": wt.num_rows, "min_df": wt["df"][-1].as_py(),
                       "what": "top lowercase letter-run words by doc frequency (regex on raw text, per-chunk df>=2)",
                       "sha256": hashlib.sha256(open(os.path.join(d, "words_df.parquet"), "rb").read()).hexdigest()}
    man["complete"] = True
    save_json(man, mpath)
    print(f"ner done: {json.dumps(man['stats'])}; {wt.num_rows:,} common words, min df {man['words_df']['min_df']}")


def rename_stage(args):
    nd = os.path.join(args.out_dir, "ner")
    ner_bytes = open(os.path.join(nd, "manifest.json"), "rb").read()
    ner = json.loads(ner_bytes)
    assert ner["complete"], "run the ner stage first"
    od = os.path.join(args.out_dir, args.arm)
    os.makedirs(od, exist_ok=True)
    mpath = os.path.join(od, "manifest.json")
    if os.path.exists(mpath) and json.load(open(mpath))["complete"]:
        print(f"{mpath} says complete; nothing to do")
        return
    # ponytail: not resumable mid-arm (minutes on a big instance); a re-run starts the arm over

    tags, swap = args.arm == "tag100", args.arm.startswith("swap")
    tab = pq.read_table(os.path.join(nd, ner["entity_df"]["file"]))
    thr, achieved = pick_threshold(tab["df"].to_numpy(), tab["mentions"].to_numpy(), ARMS[args.arm])
    kept = tab.filter(pc.greater(tab["df"], thr))
    keep = set(zip(kept["surface"].to_pylist(), kept["label"].to_pylist()))
    block, pool = set(), None
    if swap:  # real rare names (2 <= df <= threshold, so seen twice and never a kept famous name), per label
        cand = tab.filter(pc.and_(pc.greater_equal(tab["df"], 2), pc.less_equal(tab["df"], thr)))
        rng = random.Random(f"{args.arm}:pool")
        by = {}
        for surf, lab in zip(cand["surface"].to_pylist(), cand["label"].to_pylist()):
            if NAME.fullmatch(surf):
                by.setdefault(lab, []).append(tuple(surf.split()))
        pool = {}
        for lab, names in by.items():  # ponytail: capped per label so every worker's copy stays ~100 MB
            names = rng.sample(names, min(len(names), args.swap_pool))
            pool[lab] = names
            for nm in names:
                pool.setdefault((lab, len(nm)), []).append(nm)
        for lab in LABELS:  # a label with no clean names borrows the whole pool
            pool.setdefault(lab, [nm for lb in by for nm in pool[lb]])
        print("swap pool: " + ", ".join(f"{lab} {len(pool[lab]):,}" for lab in LABELS), flush=True)
    elif not tags:  # surrogates are never a common word or a word of any entity seen by NER
        ent = pc.unique(pc.list_flatten(pc.split_pattern_regex(pc.utf8_lower(tab["surface"]), r"[^\p{L}]+")))
        block = {w for w in ent.to_pylist() if len(w) > 1}
        block.update(pq.read_table(os.path.join(nd, ner["words_df"]["file"]))["word"].to_pylist())
    one_tok = {o + v + c for o in ONSETS for v in VOWELS for c in CODAS if ntok(" " + (o + v + c).capitalize()) == 1}
    one_tok = set() if swap else one_tok
    free1 = len(one_tok - block)  # if ~0, 1-token names get 2-token surrogates (token_ratio goes up)
    print(f"{free1} of {len(one_tok)} one-token syllables free for surrogates", flush=True)
    print(f"{args.arm}: df threshold {thr} -> {achieved:.4f} of mentions renamed; {len(keep):,} keys kept; "
          f"{len(block):,} blocked surrogate words", flush=True)
    val_docs = ner["settings"]["val_docs"]
    man = {"dataset": ner["dataset"], "config": ner["config"], "revision": ner["revision"],
           "source_files": ner["source_files"],
           "ner": {**ner["ner"], "manifest_sha256": hashlib.sha256(ner_bytes).hexdigest(),
                   "entity_df_sha256": ner["entity_df"]["sha256"]},
           "arm": {"name": args.arm, "kind": "typed tags LABEL_k, numbered per doc per label" if tags else
                   "real-name swap: each letter run of a renamed surface -> a word of a real rare entity name of the same "
                   "label (2 <= df <= threshold, plain capitalized 1-4 words, up to swap_pool per label), position by "
                   "position, consistent per word within a doc, rng seeded by (arm, doc index)" if swap else
                   "pseudo-word surrogates per letter run, same surface -> same surrogate within a doc, "
                   "rng seeded by (arm, doc index), token count matched where possible; never a words_df word or "
                   "a word of any NER entity",
                   "rule": "rename mentions whose (surface, label) has doc frequency <= df_threshold, then every other "
                           "whole-word case-sensitive occurrence in the doc of a renamed surface that starts uppercase "
                           "and has >= 2 letters",
                   "target_mention_frac": ARMS[args.arm], "df_threshold": thr, "achieved_mention_frac": achieved,
                   "kept_keys": len(keep), "blocked_words": len(block),
                   "free_one_token_syllables": None if tags or swap else free1,
                   "swap_pool": {lab: len(pool[lab]) for lab in LABELS} if swap else None},
           "tokenizer": "tiktoken gpt2 (encode_ordinary), <|endoftext|>=50256 prepended to each doc",
           "format": "256 x int32 header [20240520, 1, num_tokens] + uint16 tokens (llm.c)",
           "settings": {"shard_size": args.shard_size, "val_docs": val_docs, "docs": ner["stats"]["docs"]},
           "shards": [], "complete": False}

    def flush(toks, docs_started):
        idx = len(man["shards"])
        name = f"fineweb_{'val' if idx == 0 else 'train'}_{idx:06d}.bin"
        data = shard_bytes(toks)
        with open(os.path.join(od, name + ".tmp"), "wb") as f:
            f.write(data)
        os.replace(os.path.join(od, name + ".tmp"), os.path.join(od, name))
        man["shards"].append({"file": name, "num_tokens": len(toks), "sha256": hashlib.sha256(data).hexdigest(),
                              "docs_started": docs_started})
        print(f"wrote {name}: {len(toks):,} tokens, {docs_started:,} docs started", flush=True)

    st = dict.fromkeys(["mentions", "renamed", "changed_tokens", "residual", "propagated"], 0)
    parts, n, t0 = [], 0, time.time()
    docs = iter_texts(ner["revision"], [f["path"] for f in ner["source_files"]], 0, args.stream)
    with mp.Pool(args.nprocs, initializer=_init_rename, initargs=(args.arm, keep, block, pool)) as workers:
        for c in ner["chunks"]:
            c0, c1 = c["doc_start"], c["doc_end"]
            texts = list(islice(docs, c1 - c0))
            sp = pq.read_table(os.path.join(nd, c["spans"]))
            s, e, l = (sp[k].to_numpy().tolist() for k in ("start", "end", "label"))
            cut = np.searchsorted(sp["doc"].to_numpy(), np.arange(c0, c1 + 1)).tolist()
            items = [(c0 + i, t, list(zip(s[a:b], e[a:b], l[a:b])))
                     for i, (t, a, b) in enumerate(zip(texts, cut[:-1], cut[1:]))]
            for res in workers.map(_rename_batch, [items[i:i + 256] for i in range(0, len(items), 256)]):
                for d, toks, *counts in res:
                    for k, v in zip(st, counts):
                        st[k] += v
                    if d == val_docs:  # the val shard holds exactly the val docs, whole
                        flush(np.concatenate(parts), d)
                        parts, n = [], 0
                    parts.append(toks)
                    n += len(toks)
                    while d >= val_docs and n >= args.shard_size:  # llm.c: a doc that exactly fills a shard closes it
                        buf = np.concatenate(parts)
                        flush(buf[:args.shard_size], d + 1)
                        parts, n = [buf[args.shard_size:]], n - args.shard_size
            print(f"docs {c1:,} | {(c1) / (time.time() - t0):,.0f} docs/s", flush=True)
    if n:
        flush(np.concatenate(parts), ner["stats"]["docs"])
    out_tok = sum(s["num_tokens"] for s in man["shards"])
    orig_tok = ner["stats"]["val_tokens"] + ner["stats"]["train_tokens"]
    train_tok = out_tok - man["shards"][0]["num_tokens"]
    man["stats"] = {**st, "renamed_mention_frac": st["renamed"] / max(1, st["mentions"]),
                    "residual_per_renamed": st["residual"] / max(1, st["renamed"]),
                    "tokens": out_tok, "train_tokens": train_tok, "tokens_original": orig_tok,
                    "token_ratio": out_tok / orig_tok, "frac_tokens_changed": st["changed_tokens"] / out_tok}
    save_json(man, mpath)
    print(f"{args.arm} stats: {json.dumps(man['stats'])}")
    assert st["mentions"] == ner["stats"]["mentions"], "span records and entity table disagree"
    assert train_tok >= args.min_train_tokens, f"only {train_tok:,} train tokens < {args.min_train_tokens:,}"
    man["complete"] = True
    save_json(man, mpath)
    print(f"done: {len(man['shards'])} shards, {out_tok:,} tokens, {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------- self-check
def _selftest():
    def doc(*parts):  # parts: str or (str, label) -> (text, spans)
        text, spans = "", []
        for p in parts:
            if isinstance(p, tuple):
                spans.append((len(text), len(text) + len(p[0]), p[1]))
                p = p[0]
            text += p
        return text, spans

    text, spans = doc(("Marie Curie", "PERSON"), " met ", ("Pierre", "PERSON"), " in ", ("Paris", "GPE"),
                      " on 12 May 1898 (3 kg). ", ("Marie Curie", "PERSON"), "'s notes went to ", ("Paris", "GPE"),
                      " and\n", ("Paris", "PERSON"), ".")
    # tags: exact output, numbering per doc per label, numbers/dates/possessive untouched
    new, reps, ch, resid, nprop = rename_doc(text, spans, "tag100", 0)
    assert new == "PERSON_1 met PERSON_2 in GPE_1 on 12 May 1898 (3 kg). PERSON_1's notes went to GPE_1 and\n" \
                  "PERSON_3.", new
    assert resid == nprop == 0 and ch == sum(map(ntok, ["PERSON_1", " PERSON_2", " GPE_1", " PERSON_1", " GPE_1", "PERSON_3"]))

    # surrogates: only span text changes, consistency within a doc, independence across docs, determinism
    new, reps, ch, resid, nprop = rename_doc(text, spans, "r100", 7)
    assert resid == nprop == 0
    gaps = [text[a:b] for a, b in zip([0] + [e for _, e, _ in spans], [s for s, _, _ in spans] + [len(text)])]
    assert new == "".join(g + r for g, r in zip(gaps, reps + [""])), "text outside spans must be unchanged"
    assert reps[0] == reps[3] and reps[2] == reps[4] == reps[5] and len(set(reps)) == 3, reps  # keyed by surface
    assert all(r.split() != text[s:e].split() for r, (s, e, _) in zip(reps, spans))
    assert len(reps[0].split()) == 2 and all(w[0].isupper() and w[1:].islower() for w in reps[0].split())
    assert "Marie" not in new and "Curie" not in new and "Paris" not in new and "12 May 1898 (3 kg)" in new
    assert rename_doc(text, spans, "r100", 7)[0] == new, "deterministic per (arm, doc)"
    assert rename_doc(text, spans, "r100", 8)[1][0] != reps[0], "independent across docs"
    assert rename_doc(text, spans, "r50", 7)[1][0] != reps[0], "independent across arms"
    # multi-word / punctuation / case shape; keep set; blocked words
    t2, sp2 = doc("The ", ("New\nYork Times", "ORG"), " and ", ("U.S.", "GPE"), " and ", ("IBM", "ORG"), " sold ",
                  ("iPhone 12", "PRODUCT"), ".")
    n2, r2, *_ = rename_doc(t2, sp2, "r80", 1, keep={("U.S.", "GPE")}, block={"kao"})
    assert r2[1] is None and "U.S." in n2 and n2.startswith("The ") and n2.endswith(".")
    assert re.fullmatch(r"[A-Z][a-z]+\n[A-Z][a-z]+ [A-Z][a-z]+", r2[0]), r2[0]
    assert re.fullmatch(r"[A-Z]{3}", r2[2]) and r2[2] != "IBM" and re.fullmatch(r"[A-Z][a-z]+ 12", r2[3]), r2
    first_half = {a + b for a in "abcdefghijklm" for b in string.ascii_lowercase}
    assert all(pseudo_word(random.Random(i), "AB", first_half)[0] > "M" for i in range(200)), "blocked words avoided"
    # propagation: untagged whole-word, case-sensitive repeats of a renamed surface (uppercase start, >= 2 letters)
    # get the same replacement; never inside a longer word, a lowercase variant, a kept span or for 1-letter surfaces
    t3, sp3 = doc(("Curie", "PERSON"), " wrote. Curie left; Curies, curie and ", ("Curie Institute", "ORG"), ". ",
                  ("X", "ORG"), " met X.")
    kp = {("Curie Institute", "ORG")}
    n3, r3, ch3, res3, np3 = rename_doc(t3, sp3, "r100", 0, keep=kp)
    c, x = r3[0], r3[2]
    assert n3 == f"{c} wrote. {c} left; Curies, curie and Curie Institute. {x} met X.", n3
    assert np3 == 1 and res3 == 2 and ch3 == ntok(c) + ntok(" " + c) + ntok(" " + x)  # residual: kept span + "X"
    # real-name swap: originals gone, replacements from the pool, word-consistent partial mentions, case kept
    pool = {"PERSON": [("Maria", "Delgado"), ("Tom", "Hartley"), ("Karen", "Okafor"), ("Lee",)],
            ("PERSON", 2): [("Maria", "Delgado"), ("Tom", "Hartley"), ("Karen", "Okafor")], ("PERSON", 1): [("Lee",)],
            "ORG": [("Acme",), ("Globex",)], ("ORG", 1): [("Acme",), ("Globex",)]}
    t4, sp4 = doc(("Ada Lovelace", "PERSON"), " met ", ("Charles Babbage", "PERSON"), ". ", ("Lovelace", "PERSON"),
                  " joined ", ("IBM", "ORG"), "; ", ("Babbage", "PERSON"), " left.")
    n4, r4, *_ = rename_doc(t4, sp4, "swap50", 3, pool=pool)
    assert not any(w in n4 for w in ("Ada", "Lovelace", "Charles", "Babbage", "IBM")), n4
    (f1, l1), (f2, l2) = r4[0].split(), r4[1].split()
    assert r4[2] == l1 and r4[4] == l2 and l1 != l2 and f1 != f2, r4  # surnames reused, people kept distinct
    assert r4[3] in ("ACME", "GLOBEX") and {f1, l1, f2, l2} <= {w for nm in pool["PERSON"] for w in nm}, r4
    assert rename_doc(t4, sp4, "swap50", 3, pool=pool)[0] == n4 and n4.endswith(" left.")
    assert rename_doc(t3, sp3, "tag100", 0, keep=kp)[0] ==         "PERSON_1 wrote. PERSON_1 left; Curies, curie and Curie Institute. ORG_1 met X."

    # span trimming (applied at NER time) and long-doc pieces
    s = "the European Union's budget"
    assert s[slice(*trim(s, 0, len("the European Union's")))] == "European Union"
    assert trim("The Hague", 0, 9) == (4, 9) and trim("Theo", 0, 4) == (0, 4)
    long = ("word " * 30 + "\n") * 2000
    ps = list(pieces(long, 1000))
    assert "".join(p for _, p in ps) == long and all(len(p) <= 1000 for _, p in ps)
    assert all(long[o:o + len(p)] == p for o, p in ps) and all(p.startswith("\n") for _, p in ps[1:])

    # threshold: rarest keys first, closest achieved fraction, 100% -> max df
    df, m = np.array([100, 1, 5, 2, 1]), np.array([90, 2, 4, 3, 1])
    assert pick_threshold(df, m, 0.5) == (5, 0.1)  # cum at df<=1,2,5,100: .03 .06 .10 1.0
    assert pick_threshold(df, m, 0.05) == (2, 0.06) and pick_threshold(df, m, 1.0) == (100, 1.0)
    print("rename_entities selftest OK")


def default_nprocs(gb_per_proc=0.6):
    """Workers this container may use: cgroup CPU quota (v2 cpu.max, v1 cfs quota/period), else CPU affinity, else
    os.cpu_count(); capped by available RAM (min of MemAvailable and the cgroup memory limit) / gb_per_proc."""
    def read(path):
        try:
            return open(path).read().split()
        except OSError:
            return []
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    q = read("/sys/fs/cgroup/cpu.max") or read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") +         read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if len(q) == 2 and q[0] not in ("max", "-1"):
        n = min(n, max(1, int(q[0]) // int(q[1])))
    mem = [int(v[0]) for v in map(read, ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"))
           if v and v[0].isdigit()]
    mi = read("/proc/meminfo")
    if "MemAvailable:" in mi:
        mem.append(int(mi[mi.index("MemAvailable:") + 1]) * 1024)
    if mem:
        n = min(n, max(1, int(min(mem) / (gb_per_proc * 1e9))))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=["ner", "rename"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data_cache", "renamed"))
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--nprocs", type=int, default=None, help="default: CPU quota / affinity, capped by RAM / 0.6 GB")
    ap.add_argument("--stream", action="store_true", help="read parquet row groups over HTTP (tiny local tests)")
    ap.add_argument("--val_docs", type=int, default=VAL_DOCS)
    ap.add_argument("--train_tokens", type=int, default=2_600_000_000, help="ner: original train tokens to cover")
    ap.add_argument("--chunk_docs", type=int, default=100_000)
    ap.add_argument("--shard_size", type=int, default=10**8)
    ap.add_argument("--min_train_tokens", type=int, default=2_550_000_000, help="rename: fail if fewer")
    ap.add_argument("--swap_pool", type=int, default=100_000, help="swap arms: max replacement names per label")
    args = ap.parse_args()
    args.nprocs = args.nprocs or default_nprocs()
    if args.stage:
        print(f"nprocs {args.nprocs}", flush=True)
    if args.selftest:
        _selftest()
    elif args.stage == "ner":
        ner_stage(args)
    elif args.stage == "rename":
        assert args.arm, "--arm required"
        rename_stage(args)
    else:
        ap.error("give a stage or --selftest")


if __name__ == "__main__":
    main()
