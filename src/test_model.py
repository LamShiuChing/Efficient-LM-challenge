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

# 4. RoPE: q.k after rotation depends only on the position offset, and t=0 is the identity
from src.model import apply_rope, rope_cache  # noqa: E402
cos, sin = rope_cache(64, 16, 10000.0)
q, k = torch.randn(1, 1, 64, 16, dtype=torch.float64), torch.randn(1, 1, 64, 16, dtype=torch.float64)
q, k = q[:, :, :1].expand(1, 1, 64, 16), k[:, :, :1].expand(1, 1, 64, 16)  # same vector at every position
rq, rk = apply_rope(q, cos.double(), sin.double()), apply_rope(k, cos.double(), sin.double())
s = rq[0, 0] @ rk[0, 0].T  # s[i, j] = <q at i, k at j>
assert torch.allclose(s[10, 3], s[50, 43]) and torch.allclose(s[7, 7], s[0, 0]), "score must depend on i - j only"
assert torch.allclose(rq[0, 0, 0], q[0, 0, 0]) and not torch.allclose(s[10, 3], s[10, 4])

# 5. RoPE model: no position table, overfits, round-trips, handles T < block_size
rcfg = GPTConfig(block_size=32, vocab_size=128, n_layer=2, n_head=2, n_embd=64, pos="rope")
m = GPT(rcfg).to(dev)
assert "wpe" not in m.transformer and not any("rope" in k for k in m.state_dict())
assert m.num_params() == m.num_params(False) - 128 * 64
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
for i in range(50):
    _, loss = m(x[:, :-1], x[:, 1:])
    first = loss.item() if i == 0 else first
    opt.zero_grad()
    loss.backward()
    opt.step()
assert loss.item() < 0.3 * first, (first, loss.item())
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "r.pt"
    torch.save({"model": m.state_dict(), "config": dataclasses.asdict(rcfg)}, p)
    m.eval()
    assert torch.equal(m(x[:, :-1])[0], load_checkpoint(p, dev)(x[:, :-1])[0])
    assert torch.allclose(m(x[:, :8])[0], m(x[:, :-1])[0][:, :8], atol=1e-5), "causal: a prefix gives the same logits"
print(f"rope overfit: {first:.3f} -> {loss.item():.3f}")

# 6. the 1B shape (docs/PLAN_1B.md), counted on the meta device (no memory)
with torch.device("meta"):
    big = GPT(GPTConfig(block_size=4096, n_layer=18, n_head=16, n_embd=2048, pos="rope"))
assert 905e6 < big.num_params() < 907e6 and 1.008e9 < big.num_params(False) < 1.010e9, (big.num_params(), big.num_params(False))
print(f"1B shape: {big.num_params(False)/1e9:.3f}B total, {big.num_params()/1e6:.1f}M non-emb")
print("OK")
