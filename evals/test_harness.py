"""Offline check of the scoring math and item builders (no network): python -m evals.test_harness"""
import json
import re
from collections import Counter

import torch

from evals import harness
from evals.harness import POPQA_TEMPLATES, _metrics, popqa_build, score, synth

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

# Buckets: per-bucket metrics carry the same keys as the overall ones; majority baseline follows the prior.
r = _metrics("popqa_closed", [{"choices": [None, None], "label": 0, "bucket": "tail", "prior": [1, 5]}], s)
assert r["buckets"]["tail"]["acc_norm"] == 1.0 and set(r["buckets"]) == {"tail"}, r
assert r["majority_baseline"] == 0.0 == r["buckets"]["tail"]["majority_baseline"], r
assert _metrics("popqa_closed", [{"choices": [None, None], "label": 0, "bucket": "tail", "prior": [5, 1]}],
                s)["majority_baseline"] == 1.0


# PopQA builder on inline rows. Aliases of USA show up as golds of other questions and must never be
# picked; the repeated question unions its aliases (Canada is also gold there); "color" has too few objects.
def row(i, q, prop, answers, s_pop, subj=""):
    return {"id": i, "subj": subj or f"S{i}", "prop": prop, "question": q,
            "possible_answers": json.dumps(answers), "s_pop": s_pop}


rows = [row(1, "Where is A?", "country", ["United States of America", "USA", "the US"], 100, "SA"),
        row(2, "Where is A?", "country", ["Canada", "CA"], 100, "SA")]
rows += [row(10 + i, f"Where is B{i}?", "country", [g], p) for i, (g, p) in enumerate(
    [("usa", 375), ("the us", 376), ("Canada", 2879), ("France", 2880), ("Peru", 9), ("Chile", 9), ("Japan", 9)] * 3)]
rows += [row(50 + i, f"Color of C{i}?", "color", [g], 9) for i, g in enumerate(["red", "white", "black"])]
rows += [row(60, "Where is D?", "country", ["Peru"], 9), {**row(61, "Where is E?", "country", ["Peru"], 9), "subj": None}]
recs = popqa_build(rows)
assert recs == popqa_build(rows)  # deterministic
assert [x["question"] for x in recs].count("Where is A?") == 1 and not any(x["question"].startswith("Color") for x in recs)
assert not any(x["question"] == "Where is E?" for x in recs)
ali, gold_of = {}, {}
for d in rows:
    ali.setdefault(d["question"], set()).update(a.lower() for a in json.loads(d["possible_answers"]))
    gold_of.setdefault((d["subj"], d["prop"]), json.loads(d["possible_answers"])[0])  # first row wins
n_gold = Counter((x["prop"], x["answers"][x["label"]]) for x in recs)
for x in recs:
    a = [y.lower() for y in x["answers"]]
    assert len(a) == 4 and len(set(a)) == 4, x
    assert a[x["label"]] in ali[x["question"]], x
    assert not any(y in ali[x["question"]] for i, y in enumerate(a) if i != x["label"]), x
    # Oracle facts: one per choice, each about a distinct subject; gold bound to this question's subject,
    # every distractor bound to a subject whose PopQA gold it really is.
    assert sorted(o for _, o in x["facts"]) == sorted(x["answers"]) and len({s for s, _ in x["facts"]}) == 4, x
    assert (x["subj"], x["answers"][x["label"]]) in x["facts"], x
    assert all(gold_of[(s, x["prop"])] == o for s, o in x["facts"]), x
    assert not any(s.lower() in ali[x["question"]] for s, _ in x["facts"]), x  # gold never appears as a subject
    assert x["prior"] == [n_gold[(x["prop"], y)] for y in x["answers"]], x
# A row whose subject is the gold ("x" ~ "X") must not supply a distractor fact: only Y2..Y4 remain.
small = [row(80, "Q?", "genre", ["X"], 9, "Zq"), row(81, "Q1?", "genre", ["Y1"], 9, "x")]
small += [row(82 + i, f"Q{i + 2}?", "genre", [f"Y{i + 2}"], 9) for i in range(3)]
assert sorted(next(x for x in popqa_build(small) if x["question"] == "Q?")["answers"]) == ["X", "Y2", "Y3", "Y4"]
assert next(x for x in recs if x["question"] == "Where is A?")["bucket"] == "tail"
assert [x["bucket"] for x in recs[1:5]] == ["tail", "torso", "torso", "head"], [x["bucket"] for x in recs[1:5]]

# Rendered oracle prompt (no network: _load is stubbed with the inline rows).
harness._load = lambda *a, **k: rows
for split in ("dev", "test"):
    for it, cb in zip(harness.popqa(split, oracle=True), harness.popqa(split, oracle=False)):
        ctx = it["choices"][0][0]
        assert all(c[0] == ctx for c in it["choices"]) and [c[1] for c in it["choices"]] == [c[1] for c in cb["choices"]]
        assert ctx.endswith(cb["choices"][0][0]) and ctx.count(".") == 4  # 4 facts, then the closed-book prompt
        assert all(c[1].strip() in ctx for c in it["choices"]), ctx  # every choice is stated in the context
        rec = next(x for x in recs if f"Q: {x['question']}" in ctx)
        assert POPQA_TEMPLATES[rec["prop"]].format(subj=rec["subj"], obj=rec["answers"][rec["label"]]) in ctx

# Synthetic: parse each context back into facts and check the item logic.
PF = re.compile(r"(\w+) (?:works in|lives in|was born in) (\w+)\.")
CF = re.compile(r"(\w+) is a city in (\w+)\.")
for split in ("dev", "test"):
    one, two = synth(split, 1), synth(split, 2)
    assert one == synth(split, 1)
    for x1, x2 in zip(one, two):
        c1, c2 = x1["choices"][0][0], x2["choices"][0][0]
        facts, cloze = c1[:c1.rindex(".") + 1], c1[c1.rindex(".") + 2:]
        assert c2 == facts + " " + cloze + " a city in the country of"  # same context, fixed 2-hop phrasing
        sents = facts.split(". ")
        assert len(sents) == 8
        rel = PF.search(facts).group(0).split(" ", 1)[1].rsplit(" ", 1)[0]
        assert cloze.split(" ", 1)[1] == rel  # 1-hop cloze = "{person} {relation}"
        city_of, country_of = dict(PF.findall(facts)), dict(CF.findall(facts))
        assert len(city_of) == 4 and len(country_of) == 4 and set(city_of.values()) == set(country_of)
        names = list(city_of) + list(country_of) + list(country_of.values())
        assert all(a == b or a not in b for a in names for b in names), names  # distinct, no substrings
        p = cloze.split(" ", 1)[0]
        o1, o2 = [c.strip() for _, c in x1["choices"]], [c.strip() for _, c in x2["choices"]]
        assert set(o1) == set(country_of) and set(o2) == set(country_of.values())  # right type, from context
        assert o1[x1["label"]] == city_of[p] and o2[x2["label"]] == country_of[city_of[p]]
        assert all(facts.count(c) == 2 for c in o1) and all(facts.count(k) == 1 for k in o2)  # no frequency cue
        gold = o2[x2["label"]]
        assert not any(p in st and gold in st for st in sents)  # 2-hop: no single fact links person to answer
    assert one[0]["choices"][0][0] != synth("test" if split == "dev" else "dev", 1)[0]["choices"][0][0]
print("ok")
