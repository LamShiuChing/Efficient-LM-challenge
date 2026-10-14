"""Log-likelihood eval harness for base models (Phase 0). Spec: evals/README.md.

evaluate(logits_fn, encode, tasks, limit=None, split='dev') -> results dict.
  logits_fn(LongTensor[B,T]) -> FloatTensor[B,T,V]. The input is a CPU tensor; logits_fn
  moves it to its own device. Rows are right-padded with 0; causal models ignore the padding.
  encode(str) -> list[int], tiktoken gpt2.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data_cache" / "hf"))  # C: is nearly full

import numpy as np
import torch

SUITE_VERSION = "evals-v0-dev"
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


TASKS = {"hellaswag": hellaswag, "piqa": piqa, "winogrande": winogrande, "arc_easy": arc_easy,
         "lambada": lambada, "blimp": blimp}
NORM_TASKS = {"hellaswag", "piqa", "arc_easy"}  # tasks whose choices differ in length


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
    res, k = {"n": len(items)}, 0
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
    res["acc"], res["acc_ci95"] = _ci(acc)
    if task in NORM_TASKS:
        res["acc_norm"], res["acc_norm_ci95"] = _ci(acc_norm)
    if task == "lambada":  # lm-eval: ppl = exp(-mean over items of target-word logprob)
        m, (lo, hi) = _ci(ll)
        res["ppl"], res["ppl_ci95"] = float(np.exp(-m)), [float(np.exp(-hi)), float(np.exp(-lo))]
    else:  # continuous metric: mean per-token logprob of the gold continuation
        res["gold_logprob_per_tok"], res["gold_logprob_per_tok_ci95"] = _ci(gold_lp)
    return res


def evaluate(logits_fn, encode, tasks, limit=None, split="dev", batch_size=32, max_len=1024):
    assert split in ("dev", "test"), split
    results = {"suite_version": SUITE_VERSION, "split": split, "limit": limit, "tasks": {}}
    for task in tasks:
        items = TASKS[task](split)[:limit]
        reqs = [(encode(c), encode(t)) for it in items for c, t in it["choices"]]
        results["tasks"][task] = _metrics(task, items, score(logits_fn, reqs, batch_size, max_len))
    return results
