"""Offline check of the scoring math (no network): python -m evals.test_harness"""
import torch

from evals.harness import _metrics, score

# Fake bigram LM over 8 tokens: logits at a position depend only on the current token.
V = 8
TABLE = torch.full((V, V), -10.0)
TABLE[1, 2] = TABLE[2, 3] = TABLE[3, 4] = 3.0  # 1 -> 2 -> 3 -> 4 is the greedy path
TABLE[4, 4], TABLE[4, 6] = 0.0, -0.2  # 4 -> 4 is greedy but costs ~0.6 nats
TABLE[1, 5] = 2.5
LP = TABLE.log_softmax(-1)


def logits_fn(x):
    return TABLE[x.clamp(max=V - 1)]  # EOT (50256) maps to token 7


def lp(path):
    return sum(LP[a, b].item() for a, b in zip(path, path[1:]))


# Exact sums, token counts, greedy flags; mixed lengths in one batch exercise right-padding.
E = ([1], [2, 3, 4, 4, 4, 4])  # long, cheap per token
F = ([1], [5])                 # short, one costlier token
s = score(logits_fn, [E, F], batch_size=2)
assert abs(s[0][0] - lp([1, 2, 3, 4, 4, 4, 4])) < 1e-4 and abs(s[1][0] - lp([1, 5])) < 1e-4, s
assert [x[1] for x in s] == [6, 1] and s[0][2] and not s[1][2]

# Sum vs length-normalized pick different options; gold is E, so acc=0 and acc_norm=1.
assert s[1][0] > s[0][0] and s[0][0] / 6 > s[1][0] / 1, s
r = _metrics("hellaswag", [{"choices": [None, None], "label": 0}], s)
assert r["acc"] == 0.0 and r["acc_norm"] == 1.0, r

# Empty context is scored after EOT; left-truncation keeps the continuation intact.
assert abs(score(logits_fn, [([], [1])])[0][0] - LP[7, 1].item()) < 1e-4
t = score(logits_fn, [([0] * 50 + [1], [2, 3])], max_len=4)[0]
assert abs(t[0] - lp([1, 2, 3])) < 1e-4 and t[2], t
print("ok")
