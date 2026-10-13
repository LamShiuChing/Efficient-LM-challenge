# NNN — short name

> Copy to `experiments/NNN-short-name.md`. Fill every section above the line **before** launching and commit it. See [`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md).

- **Status:** preregistered | running | done | abandoned | exploratory
- **Phase / track:** e.g. Phase 1 / Track 1
- **Date preregistered:** YYYY-MM-DD
- **Spec commit:** (hash of the commit that adds this file)

## Question

One sentence. What will we know after this run that we don't know now?

## Hypothesis

What you expect and why. Link the plan or literature section it comes from.

## Prediction

Concrete, falsifiable numbers. Example: "knowledge-light 125M is within the noise floor of normal 125M on oracle-context HotpotQA, and ≥ 5 points lower on closed-book TriviaQA."

## Decision rule

What result leads to what next step. Include the null case.

| Outcome | Interpretation | Next |
|---|---|---|
| | | |

## The one change

- **Varied factor:** (exactly one)
- **Arms:** A = …, B = …
- **Held fixed:** tokenizer, token budget, schedule, eval suite version, seeds, …

## Config

- Model: layers / d_model / heads / context, params (total, non-embedding)
- Data: recipe + filter version + manifest hash
- Tokens: …
- Optimizer / LR / schedule / batch size
- Seeds: …
- Eval suite version: `evals-vN`; tasks and conditions (closed-book / oracle-context / retrieved)

## Budget

- Estimated FLOPs (`6ND`) and GPU-hours per arm
- Hardware: local 4080 SUPER | rented (type, $/h)
- Hard cap: stop and report if exceeding …

---

*Everything below is filled in after the run. Do not edit anything above this line.*

## Deviations

Anything that differed from the spec, and why.

## Results

Tables with mean ± std over seeds and bootstrap CI. Noise floor for reference. Links to logs and checkpoints.

## Conclusion

Which row of the decision rule happened. What it means for the plan. Put surprises and negative results here too.
