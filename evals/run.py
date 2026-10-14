"""python -m evals.run --model hf:gpt2 --tasks hellaswag,piqa [--split dev] [--limit N] --out results.json"""
import argparse
import json
import time

from evals import adapters
from evals.harness import TASKS, evaluate

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="hf:<name> or ckpt:<path>")
p.add_argument("--tasks", default=",".join(TASKS))
p.add_argument("--split", default="dev", choices=["dev", "test"])
p.add_argument("--limit", type=int)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--out")
a = p.parse_args()

kind, _, arg = a.model.partition(":")
logits_fn, encode = {"hf": adapters.hf_gpt2, "ckpt": adapters.local_ckpt}[kind](arg)
res = {"model": a.model, "split": a.split, "tasks": {}}
for t in a.tasks.split(","):
    t0 = time.time()
    r = evaluate(logits_fn, encode, [t], a.limit, a.split, a.batch_size)
    r["tasks"][t]["seconds"] = round(time.time() - t0, 1)
    res["suite_version"] = r["suite_version"]
    res["tasks"][t] = r["tasks"][t]
    print(t, json.dumps(r["tasks"][t]), flush=True)
res["limit"] = a.limit
if a.out:
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
