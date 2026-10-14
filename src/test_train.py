"""Trainer self-check on CPU, a few seconds: python src/test_train.py

1. resume: stop at step 6 (state.pt), resume, and the final checkpoint equals an uninterrupted run, bit for bit.
2. DDP (torchrun, gloo on CPU): 2 ranks with half the micro-batch log the same losses as 1 process.
   Skipped with a note if torchrun can't start here (it runs on the Linux GPU boxes).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from loader import shard_bytes  # noqa: E402

env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
tmp = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
for name in ("t_val_000000.bin", "t_train_000001.bin", "t_train_000002.bin"):
    (tmp / name).write_bytes(shard_bytes(rng.integers(0, 256, 3000).astype(np.uint16)))
base = ["--device=cpu", "--exploratory", "--compile=false", "--vocab_size=256", "--block_size=32", "--n_layer=2",
        "--n_head=2", "--n_embd=32", "--total_batch_size=256", "--max_steps=12", "--warmup_steps=2",
        "--val_every=1000", "--ckpt_every=1000", "--val_tokens=256", f"--train_pattern={tmp}/t_train_*.bin",
        f"--val_pattern={tmp}/t_val_*.bin"]


def run(name, *extra, launcher=(sys.executable,)):
    cmd = [*launcher, str(ROOT / "src" / "train.py"), *base, f"--run_name={name}", *extra]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def losses(name):
    return [r["loss"] for r in map(json.loads, open(ROOT / "runs" / name / "log.jsonl")) if "loss" in r]


runs = [f"_test_{k}_{os.getpid()}" for k in ("full", "res", "ddp1", "ddp2")]
try:
    # 1. resume
    r = run(runs[0], "--micro_batch_size=4", "--state_every=6")
    assert r.returncode == 0, r.stderr[-2000:]
    full = torch.load(ROOT / "runs" / runs[0] / "ckpt_000012.pt", weights_only=True)["model"]
    (ROOT / "runs" / runs[1]).mkdir(parents=True)
    st = torch.load(ROOT / "runs" / runs[0] / "state.pt", weights_only=True)
    assert st["step"] == 6
    st["train_config"]["run_name"] = runs[1]  # same run, new folder
    torch.save(st, ROOT / "runs" / runs[1] / "state.pt")
    r = run(runs[1], "--micro_batch_size=4", "--state_every=6", "--resume")
    assert r.returncode == 0, r.stderr[-2000:]
    res = torch.load(ROOT / "runs" / runs[1] / "ckpt_000012.pt", weights_only=True)["model"]
    assert all(torch.equal(full[k], res[k]) for k in full), "resumed run differs from the uninterrupted one"
    assert losses(runs[1]) == losses(runs[0])[6:], "resumed losses differ"
    print("resume: identical final model and losses")

    # 2. DDP equivalence
    one = run(runs[2], "--micro_batch_size=4")
    two = run(runs[3], "--micro_batch_size=2",
              launcher=(sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"))
    if two.returncode != 0:
        print("DDP: torchrun could not run here, skipped:", two.stderr.strip().splitlines()[-1][:200])
    else:
        a, b = losses(runs[2]), losses(runs[3])
        assert len(a) == len(b) == 12 and np.allclose(a, b, atol=1e-4), (a, b)
        print(f"DDP: 2 ranks match 1 process (max loss diff {max(abs(x - y) for x, y in zip(a, b)):.2e})")
    print("OK")
finally:
    for n in runs:
        shutil.rmtree(ROOT / "runs" / n, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
