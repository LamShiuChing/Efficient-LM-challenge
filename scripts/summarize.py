"""Results table for runs on the HF runs repo (reads JSON only; no compute).

    python scripts/summarize.py e001_gpt2_124m_2p5B_s1337 e001_gpt2_124m_2p5B_s1338 ... [--split test] [--ref gpt2_test]

Prints per-run final val loss + eval metrics, and mean / std over the runs given.
"""
import argparse
import json
import statistics as st

from huggingface_hub import hf_hub_download

REPO = "lamshiu/efficient-lm-runs"


def get(path):
    try:
        return hf_hub_download(REPO, path)
    except Exception:
        return None


def final_val_loss(run):
    p = get(f"runs/{run}/log.jsonl")
    if not p:
        return None
    vals = [json.loads(l) for l in open(p, encoding="utf-8") if '"val_loss"' in l]
    return vals[-1]["val_loss"] if vals else None


def metrics(name, split):
    p = get(f"evals/{name}_{split}.json") or get(f"evals/{name}.json")
    if not p:
        return {}
    out = {}
    for task, r in json.load(open(p, encoding="utf-8"))["tasks"].items():
        out[f"{task}.acc"] = r["acc"]
        if "acc_norm" in r:
            out[f"{task}.acc_norm"] = r["acc_norm"]
        if "gold_logprob_per_tok" in r:
            out[f"{task}.gold_lp"] = r["gold_logprob_per_tok"]
        if "ppl" in r:
            out[f"{task}.ppl"] = r["ppl"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ref", help="eval file stem of a reference model, e.g. gpt2_test")
    a = ap.parse_args()
    rows = {r: {"val_loss": final_val_loss(r), **metrics(r, a.split)} for r in a.runs}
    ref = metrics(a.ref, "") if a.ref else {}
    keys = sorted({k for v in rows.values() for k in v})
    print(f"{'metric':28s}" + "".join(f"{r[-6:]:>10s}" for r in a.runs) + f"{'mean':>10s}{'std':>9s}" + (f"{'ref':>10s}" if ref else ""))
    for k in keys:
        xs = [rows[r].get(k) for r in a.runs]
        got = [x for x in xs if x is not None]
        line = f"{k:28s}" + "".join(f"{x:10.4f}" if x is not None else f"{'-':>10s}" for x in xs)
        line += f"{st.mean(got):10.4f}" if got else f"{'-':>10s}"
        line += f"{st.stdev(got):9.4f}" if len(got) > 1 else f"{'-':>9s}"
        if ref:
            line += f"{ref[k]:10.4f}" if k in ref else f"{'-':>10s}"
        print(line)


if __name__ == "__main__":
    main()
