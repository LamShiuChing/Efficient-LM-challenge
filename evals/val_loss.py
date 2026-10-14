"""Mean next-token loss of a checkpoint on llm.c-format shards (e.g. the normal FineWeb val shard),
so arms trained on different data can be compared on the same text.

    python -m evals.val_loss <ckpt.pt> <shard_glob> [--tokens 10485760] [--out x.json]
"""
import argparse
import json

import torch

from data.loader import ShardLoader
from src.model import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("pattern")
ap.add_argument("--tokens", type=int, default=10_485_760)  # same as training's val_tokens
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--out")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = load_checkpoint(a.ckpt, dev)
T = model.config.block_size
loader = ShardLoader(a.pattern, a.B, T)
steps = max(1, a.tokens // (a.B * T))
total = 0.0
with torch.no_grad():
    for _ in range(steps):
        x, y = loader.next_batch()
        with torch.autocast(dev, dtype=torch.bfloat16):
            total += model(x.to(dev), y.to(dev))[1].item()
res = {"ckpt": a.ckpt, "pattern": a.pattern, "tokens": steps * a.B * T, "val_loss": total / steps}
print(json.dumps(res))
if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
