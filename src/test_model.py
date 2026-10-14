"""Model self-check: python src/test_model.py"""
import dataclasses
import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import GPT, GPTConfig, load_checkpoint  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# 1. shapes + init loss on the real 124M shape
m = GPT(GPTConfig()).to(dev)
x = torch.randint(0, 50257, (2, 64), device=dev)
logits, loss = m(x, torch.randint(0, 50257, (2, 64), device=dev))
assert logits.shape == (2, 64, 50304), logits.shape
assert m(x)[1] is None
assert abs(loss.item() - math.log(50257)) < 0.2, loss.item()
assert 85.0e6 < m.num_params() < 85.1e6 and 124.4e6 < m.num_params(False) < 124.5e6, (m.num_params(), m.num_params(False))
print(f"init loss {loss.item():.3f} (ln 50257 = {math.log(50257):.3f}); params {m.num_params(False)/1e6:.2f}M total, "
      f"{m.num_params()/1e6:.2f}M non-emb")
del m

# 2. tiny model overfits one fixed batch
cfg = GPTConfig(block_size=32, vocab_size=128, n_layer=2, n_head=2, n_embd=64)
m = GPT(cfg).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
x = torch.randint(0, 128, (8, 33), device=dev)
for i in range(50):
    _, loss = m(x[:, :-1], x[:, 1:])
    if i == 0:
        first = loss.item()
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"overfit: {first:.3f} -> {loss.item():.3f}")
assert loss.item() < 0.3 * first

# 3. checkpoint round-trip (with a compiled-style prefix)
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "c.pt"
    sd = {"_orig_mod." + k: v for k, v in m.state_dict().items()}
    torch.save({"model": sd, "config": dataclasses.asdict(cfg), "step": 50, "tokens_seen": 0, "train_config": {}}, p)
    m2 = load_checkpoint(p, dev)
    m.eval()
    assert torch.equal(m(x[:, :-1])[0], m2(x[:, :-1])[0])
print("OK")
