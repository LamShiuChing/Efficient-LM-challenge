"""Log-likelihood eval harness for base models (Phase 0). Spec: evals/README.md.

evaluate(logits_fn, encode, tasks, limit=None, split='dev') -> results dict.
  logits_fn(LongTensor[B,T]) -> FloatTensor[B,T,V]. The input is a CPU tensor; logits_fn
  moves it to its own device. Rows are right-padded with 0; causal models ignore the padding.
  encode(str) -> list[int], tiktoken gpt2.
"""
import json
import os
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data_cache" / "hf"))  # C: is nearly full

import numpy as np
import torch

SUITE_VERSION = "evals-v0.3-dev"  # v0.3 adds the renamed PopQA variants; v0.2 tasks unchanged
EOT = 50256  # gpt2 <|endoftext|>, used as the context when a task has none (BLiMP)
SEED = 0     # dev carve-outs and bootstrap

# Dataset pins (HF repo, config, commit). Verified on the hub 2026-10-13.
REV = {
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "baber/piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "EleutherAI/lambada_openai": "900124bf3b8235c6daf21033af9948b3f07346c4",
    "cimec/lambada": "5953bd97664b64b95754f299b2309ecfbfbe81b9",
    "nyu-mll/blimp": "877fba0801ffb7cbd8c39c1ff314a46f053f6036",
    "akariasai/PopQA": "098765c79ea10a2cb19c828324e33281b8336ec0",  # verified 2026-10-14
}


def _load(repo, split, config=None):
    from datasets import load_dataset
    return load_dataset(repo, config, split=split, revision=REV[repo])


def _carve(ds, n):
    """Fixed random subset of n rows (seeded), used to make dev sets out of train splits."""
    idx = np.random.default_rng(SEED).permutation(len(ds))[:n]
    return ds.select(sorted(idx.tolist()))


# Each loader returns a list of items: {"choices": [(ctx, cont), ...], "label": int}.
# Continuations carry their leading space; ctx and cont are tokenized separately (as llm.c).

def hellaswag(split):
    # llm.c dev/data/hellaswag.py: raw ctx, " " + ending, no text preprocessing.
    ds = _load("Rowan/hellaswag", "validation") if split == "test" else _carve(_load("Rowan/hellaswag", "train"), 10042)
    return [{"choices": [(d["ctx"], " " + e) for e in d["endings"]], "label": int(d["label"])} for d in ds]


def piqa(split):
    ds = _load("baber/piqa", "validation") if split == "test" else _carve(_load("baber/piqa", "train"), 5000)
    return [{"choices": [(f"Question: {d['goal']}\nAnswer:", " " + s) for s in (d["sol1"], d["sol2"])],
             "label": int(d["label"])} for d in ds]


def winogrande(split):
    # Partial scoring (lm-eval): context = sentence up to the blank with the option filled in,
    # continuation = the rest of the sentence; identical across options.
    ds = (_load("allenai/winogrande", "validation", "winogrande_xl") if split == "test"
          else _carve(_load("allenai/winogrande", "train", "winogrande_xl"), 5000))
    items = []
    for d in ds:
        i = d["sentence"].index("_")
        cont = " " + d["sentence"][i + 1:].strip()
        items.append({"choices": [(d["sentence"][:i] + o, cont) for o in (d["option1"], d["option2"])],
                      "label": int(d["answer"]) - 1})
    return items


def arc_easy(split):
    from datasets import concatenate_datasets
    if split == "test":
        ds = _load("allenai/ai2_arc", "test", "ARC-Easy")
    else:
        ds = concatenate_datasets([_load("allenai/ai2_arc", s, "ARC-Easy") for s in ("train", "validation")])
    return [{"choices": [(f"Question: {d['question']}\nAnswer:", " " + t) for t in d["choices"]["text"]],
             "label": d["choices"]["label"].index(d["answerKey"])} for d in ds]


def lambada(split):
    # Target = last space-separated word (lm-eval lambada_openai). Single "choice", label 0.
    # dev is the original LAMBADA dev set, which is lowercased and PTB-tokenized: dev numbers
    # are not on the same scale as test, compare models on one split only.
    ds = _load("EleutherAI/lambada_openai", "test", "default") if split == "test" else _load("cimec/lambada", "validation")
    items = []
    for d in ds:
        ctx, word = d["text"].rsplit(" ", 1)
        items.append({"choices": [(ctx, " " + word)], "label": 0})
    return items


def blimp(split):
    # 67 paradigms x 1000 minimal pairs, one HF split. Seeded 500/500 dev/test per paradigm.
    from huggingface_hub import list_repo_tree
    configs = sorted(f.path for f in list_repo_tree("nyu-mll/blimp", repo_type="dataset", revision=REV["nyu-mll/blimp"])
                     if "." not in f.path)
    items = []
    for c in configs:
        ds = _load("nyu-mll/blimp", "train", c)
        perm = np.random.default_rng(SEED).permutation(len(ds))
        half = perm[:len(ds) // 2] if split == "dev" else perm[len(ds) // 2:]
        for d in ds.select(sorted(half.tolist())):
            items.append({"choices": [("", d["sentence_good"]), ("", d["sentence_bad"])], "label": 0})
    return items


# --- Phase 1a knowledge tasks -------------------------------------------------------------
# PopQA (Mallen et al. 2023): one HF split "test", 14267 rows, 16 relations (prop).
POPQA_POP_BOUNDS = (375, 2879)  # s_pop terciles (np.quantile 1/3, 2/3) over all 14267 rows
POPQA_BUCKETS = ("tail", "torso", "head")  # s_pop <= 375 | <= 2879 | > 2879
POPQA_TEMPLATES = {  # oracle-context fact, one per relation
    "occupation": "{subj}'s occupation is {obj}.",
    "place of birth": "{subj} was born in {obj}.",
    "genre": "The genre of {subj} is {obj}.",
    "father": "The father of {subj} is {obj}.",
    "mother": "The mother of {subj} is {obj}.",
    "country": "{subj} is located in {obj}.",
    "producer": "{subj} was produced by {obj}.",
    "director": "{subj} was directed by {obj}.",
    "screenwriter": "The screenwriter of {subj} was {obj}.",
    "composer": "The composer of {subj} was {obj}.",
    "author": "The author of {subj} is {obj}.",
    "capital": "The capital of {subj} is {obj}.",
    "capital of": "{subj} is the capital of {obj}.",
    "color": "The color of {subj} is {obj}.",
    "religion": "The religion of {subj} is {obj}.",
    "sport": "{subj} plays {obj}.",
}


def popqa_build(rows):
    """PopQA rows (dicts with id, subj, prop, question, possible_answers (JSON str), s_pop) -> records.

    Dedupes repeated questions (same subj+prop; first row wins, aliases are unioned). Gold = first
    possible answer. 3 distractors = golds of other questions with the same prop, sampled by row so
    frequent objects are drawn as often as they are gold; never an alias of the gold (case-insensitive),
    never repeated. The row a distractor was drawn from supplies a true fact for it, so the oracle
    context states all 4 choices, each bound to its own subject. Seeded per row id, so records do not
    depend on split or limit. prior[j] = how often answers[j] is gold in this relation (majority baseline).
    """
    recs, bad = {}, {}
    for d in rows:
        if not d["subj"]:  # 3 rows have no subject
            continue
        pa = json.loads(d["possible_answers"])
        bad.setdefault(d["question"], set()).update(a.lower() for a in pa)
        recs.setdefault(d["question"], {**d, "gold": pa[0]})
    pool = {}
    for r in recs.values():
        pool.setdefault(r["prop"], []).append(r)
    freq = {p: Counter(x["gold"] for x in v) for p, v in pool.items()}
    out = []
    for q, r in recs.items():
        rng = np.random.default_rng([SEED, int(r["id"])])
        cands, seen, dis = pool[r["prop"]], set(bad[q]), []
        for i in rng.permutation(len(cands)):
            c = cands[i]
            # its fact's subject must not be our subject or a gold alias (gold would appear twice in context)
            if c["gold"].lower() not in seen and c["subj"] != r["subj"] and c["subj"].lower() not in bad[q]:
                seen.add(c["gold"].lower())
                dis.append(c)
                if len(dis) == 3:
                    break
        if len(dis) < 3:  # relation too small to find 3 wrong objects
            continue
        lab = int(rng.integers(4))
        dis.insert(lab, r)
        facts = [(dis[j]["subj"], dis[j]["gold"]) for j in rng.permutation(4)]
        b = 0 if r["s_pop"] <= POPQA_POP_BOUNDS[0] else 1 if r["s_pop"] <= POPQA_POP_BOUNDS[1] else 2
        out.append({"question": q, "subj": r["subj"], "prop": r["prop"], "answers": [x["gold"] for x in dis],
                    "label": lab, "bucket": POPQA_BUCKETS[b], "facts": facts,
                    "prior": [freq[r["prop"]][x["gold"]] for x in dis]})
    return out


# Renamed variants (experiment 003): NER label of (subject, object) per relation, for typed tags. Object None =
# a common noun (occupation, genre, ...) that training-time NER would not tag, so it is kept.
POPQA_ENT = {
    "occupation": ("PERSON", None), "place of birth": ("PERSON", "GPE"), "genre": ("WORK_OF_ART", None),
    "father": ("PERSON", "PERSON"), "mother": ("PERSON", "PERSON"), "country": ("LOC", "GPE"),
    "producer": ("WORK_OF_ART", "PERSON"), "director": ("WORK_OF_ART", "PERSON"),
    "screenwriter": ("WORK_OF_ART", "PERSON"), "composer": ("WORK_OF_ART", "PERSON"),
    "author": ("WORK_OF_ART", "PERSON"), "capital": ("GPE", "GPE"), "capital of": ("GPE", "GPE"),
    "color": ("PRODUCT", None), "religion": ("PERSON", None), "sport": ("PERSON", None),
}


def rename_item(texts, names, arm, idx):
    """Rename names {surface: label} in the strings of one item exactly as data/rename_entities.py renamed a
    training doc (arm "r100": pseudo-word surrogates, "tag100": typed tags). The item is one doc, so a name gets
    the same replacement in context, question and choices. Longer names win where names overlap."""
    import re
    import sys
    sys.path.insert(0, str(ROOT / "data"))
    from rename_entities import rename_doc
    names = {re.sub(r"^[Tt]he\s+", "", n): lab for n, lab in names.items()}  # training spans drop a leading "the"
    text, spans = "\x00".join(texts), []
    for name in sorted(names, key=len, reverse=True):
        for m in re.finditer(r"(?<!\w)%s(?!\w)" % re.escape(name), text):
            if all(m.end() <= s or m.start() >= e for s, e, _ in spans):
                spans.append((m.start(), m.end(), names[name]))
    return rename_doc(text, sorted(spans), arm, idx)[0].split("\x00")


def popqa(split, oracle, rename=None):
    # No official dev split: seeded 50/50 partition of the deduped questions. Items stay in permuted
    # order, so --limit takes a random sample instead of the first relation in the file.
    recs = popqa_build(_load("akariasai/PopQA", "test"))
    perm = np.random.default_rng(SEED).permutation(len(recs))
    half = perm[:len(recs) // 2] if split == "dev" else perm[len(recs) // 2:]
    items = []
    for i in half:
        r = recs[i]
        ctx = (" ".join(POPQA_TEMPLATES[r["prop"]].format(subj=s, obj=o) for s, o in r["facts"]) + "\n") if oracle else ""
        q, ans = r["question"], [" " + a for a in r["answers"]]
        if rename:  # every fact subject, and the objects when they are entities; never the "Q:"/"A:" scaffold
            sl, ol = POPQA_ENT[r["prop"]]
            names = {**({a: ol for a in r["answers"]} if ol else {}), **{s: sl for s, _ in r["facts"]}}
            ctx, q, *ans = rename_item([ctx, q, *ans], names, rename, int(i))
        items.append({"choices": [(ctx + f"Q: {q}\nA:", a) for a in ans], "label": r["label"], "bucket": r["bucket"], "prior": r["prior"]})
    return items


# Synthetic in-context reasoning over fictional entities. Independent of any training-data renaming.
_ON = "b d f g k l m n p r s t v z br dr gr kr st tr th sh".split()
_VO = "a e i o u ai ea ou".split()
_CO = ["", "", "n", "r", "l", "s", "k", "m"]
SYNTH_RELS = ["works in", "lives in", "was born in"]  # person -> city relation, one per item
# Cloze prompts: 1-hop "{P} {rel}" -> city, 2-hop "{P} {rel} a city in the country of" -> country.
SYNTH_N = 1000  # items per split


def _pseudo(rng, used, country=False):
    """Pronounceable 2-syllable name; no name in one item is a substring of another."""
    while True:
        n = "".join(rng.choice(_ON) + rng.choice(_VO) + rng.choice(_CO) for _ in range(2)).capitalize()
        if country:
            n += "nia" if n[-1] in "aeiou" else "ia"
        if not any(n in u or u in n for u in used):
            used.append(n)
            return n


def synth(split, hops):
    """4 chains person -> city -> country as 8 shuffled facts, then a cloze about one person.

    1-hop completes the city (choices: the 4 cities), 2-hop the country (choices: the 4 countries).
    Every candidate appears equally often in the context, so frequency gives no hint. Item i of a
    split has the same context and person for both hop counts (rng use does not depend on hops).
    """
    items = []
    for i in range(SYNTH_N):
        rng = random.Random(f"synth-{split}-{i}")
        used = []
        P = [_pseudo(rng, used) for _ in range(4)]
        C = [_pseudo(rng, used) for _ in range(4)]
        K = [_pseudo(rng, used, country=True) for _ in range(4)]
        rel = rng.choice(SYNTH_RELS)
        facts = [f"{p} {rel} {c}." for p, c in zip(P, C)] + [f"{c} is a city in {k}." for c, k in zip(C, K)]
        rng.shuffle(facts)
        q = rng.randrange(4)
        ctx = " ".join(facts) + f" {P[q]} {rel}" + ("" if hops == 1 else " a city in the country of")
        items.append({"choices": [(ctx, " " + e) for e in (C if hops == 1 else K)], "label": q})
    return items


TASKS = {"hellaswag": hellaswag, "piqa": piqa, "winogrande": winogrande, "arc_easy": arc_easy,
         "lambada": lambada, "blimp": blimp,
         "popqa_closed": lambda s: popqa(s, oracle=False), "popqa_oracle": lambda s: popqa(s, oracle=True),
         "popqa_oracle_surr": lambda s: popqa(s, True, "r100"), "popqa_oracle_tag": lambda s: popqa(s, True, "tag100"),
         "popqa_closed_surr": lambda s: popqa(s, False, "r100"),
         "synth_1hop": lambda s: synth(s, 1), "synth_2hop": lambda s: synth(s, 2)}
# tasks whose choices differ in length
NORM_TASKS = {"hellaswag", "piqa", "arc_easy", "popqa_closed", "popqa_oracle", "popqa_oracle_surr", "popqa_oracle_tag",
              "popqa_closed_surr", "synth_1hop", "synth_2hop"}


@torch.no_grad()
def score(logits_fn, reqs, batch_size=32, max_len=1024):
    """reqs: list of (ctx_ids, cont_ids). Returns (sum logprob of cont, n cont tokens, cont is greedy)."""
    rows = [((c or [EOT]) + t)[-(max_len + 1):] for c, t in reqs]  # left-truncate to the model window
    ncont = [len(t) for _, t in reqs]
    assert min(ncont) > 0 and all(n < len(r) for n, r in zip(ncont, rows)), "empty or over-long continuation"
    out = [None] * len(reqs)
    order = sorted(range(len(reqs)), key=lambda i: -len(rows[i]))  # longest first: OOM shows up early
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        T = len(rows[idx[0]]) - 1
        x = torch.zeros(len(idx), T, dtype=torch.long)
        y = torch.zeros(len(idx), T, dtype=torch.long)
        m = torch.zeros(len(idx), T, dtype=torch.bool)
        for j, i in enumerate(idx):
            r, L = rows[i], len(rows[i]) - 1
            x[j, :L] = torch.tensor(r[:-1])
            y[j, :L] = torch.tensor(r[1:])
            m[j, L - ncont[i]:L] = True
        logits = logits_fn(x).float()
        y, m = y.to(logits.device), m.to(logits.device)
        tok_lp = logits.gather(-1, y[..., None])[..., 0] - logits.logsumexp(-1)
        lp = (tok_lp * m).sum(1).tolist()
        greedy = ((logits.argmax(-1) == y) | ~m).all(1).tolist()
        for j, i in enumerate(idx):
            out[i] = (lp[j], ncont[i], greedy[j])
    return out


def _ci(v, n_boot=1000):
    """Mean and bootstrap 95% CI over items."""
    v = np.asarray(v, dtype=np.float64)
    boots = v[np.random.default_rng(SEED).integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def _metrics(task, items, scores):
    k = 0
    acc, acc_norm, gold_lp, ll = [], [], [], []
    for it in items:
        s = scores[k:k + len(it["choices"])]
        k += len(it["choices"])
        lab = it["label"]
        if task == "lambada":
            acc.append(s[0][2])
            ll.append(s[0][0])
            continue
        acc.append(int(np.argmax([x[0] for x in s])) == lab)
        acc_norm.append(int(np.argmax([x[0] / x[1] for x in s])) == lab)
        gold_lp.append(s[lab][0] / s[lab][1])

    def summ(sel):
        pick = lambda v: [v[i] for i in sel]  # noqa: E731
        res = {"n": len(sel)}
        res["acc"], res["acc_ci95"] = _ci(pick(acc))
        if task in NORM_TASKS:
            res["acc_norm"], res["acc_norm_ci95"] = _ci(pick(acc_norm))
        if task == "lambada":  # lm-eval: ppl = exp(-mean over items of target-word logprob)
            m, (lo, hi) = _ci(pick(ll))
            res["ppl"], res["ppl_ci95"] = float(np.exp(-m)), [float(np.exp(-hi)), float(np.exp(-lo))]
        else:  # continuous metric: mean per-token logprob of the gold continuation
            res["gold_logprob_per_tok"], res["gold_logprob_per_tok_ci95"] = _ci(pick(gold_lp))
        if "prior" in items[0]:  # popqa: always pick the choice most often gold for its relation
            res["majority_baseline"] = float(np.mean([int(np.argmax(items[i]["prior"])) == items[i]["label"]
                                                      for i in sel]))
        return res

    res = summ(range(len(items)))
    if items and "bucket" in items[0]:  # popqa: the same metrics per subject-popularity bucket
        sel = {b: [i for i, it in enumerate(items) if it["bucket"] == b] for b in POPQA_BUCKETS}
        res["buckets"] = {b: summ(v) for b, v in sel.items() if v}
    return res


def evaluate(logits_fn, encode, tasks, limit=None, split="dev", batch_size=32, max_len=1024):
    assert split in ("dev", "test"), split
    results = {"suite_version": SUITE_VERSION, "split": split, "limit": limit, "tasks": {}}
    for task in tasks:
        items = TASKS[task](split)[:limit]
        reqs = [(encode(c), encode(t)) for it in items for c, t in it["choices"]]
        results["tasks"][task] = _metrics(task, items, score(logits_fn, reqs, batch_size, max_len))
    return results
