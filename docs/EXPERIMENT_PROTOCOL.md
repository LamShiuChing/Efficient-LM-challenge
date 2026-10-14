# Experiment Protocol

Rules that make results in this repo trustworthy. A result that breaks one of these rules is not reported as a finding. At most it is a lead to re-run properly.

## 1. Preregister before you run

1. Copy `experiments/_TEMPLATE.md` to `experiments/NNN-short-name.md`. `NNN` is the next free number.
2. Before launching, fill in **Hypothesis**, **Prediction**, **Decision rule**, **Config** and **Budget**.
3. Commit the spec. The commit hash of the spec is the run's preregistration.
4. After the run, fill in **Results** and **Conclusion** below the frozen sections. Never edit the frozen sections afterwards. If the plan changed mid-run, say so in **Deviations**.

A run without a committed spec is exploratory. Label it `exploratory` and do not cite it as evidence.

## 2. Change one thing at a time

- A comparison changes exactly one factor: data recipe, model size, or training method. Never change data and model together.
- Everything else stays fixed: tokenizer, token budget, LR schedule shape, eval suite version, seed set.
- If two factors must move together (e.g. LR re-tuned for a new size), the spec says so and says how the second factor was set. Example: "LR from the Phase 0 sweep rule, not tuned on the eval".

## 3. Frozen eval suite

- The eval suite is frozen at the end of Phase 0 and tagged `evals-v1`.
- Any change to the suite (new task, new prompt format, new scoring) bumps the version. Every baseline that is compared against must be re-scored on the new version.
- Every eval report names the suite version.

## 4. Data splits and contamination

- Each eval set has a **dev** split, used freely during development, and a **test** split. The test split is touched only for the final numbers of a preregistered comparison and is never used for any choice: not the checkpoint, not the hyperparameters, not the prompt.
- Training data is decontaminated against all eval sets (n-gram overlap; the method and threshold go in `evals/README.md` once written). Report the decontamination method with results.
- Retrieval corpora must not contain eval questions paired with their answers (e.g. trivia-site Q&A dumps). Check before using a corpus.

## 5. Noise floor and statistics

- Phase 0 measures the **noise floor**: the seed-to-seed std of every eval metric at the baseline size, from at least 3 seeds.
- A difference smaller than **2× the seed std** is reported as "no detectable difference". It is never reported as a win or a loss.
- Report mean ± std over seeds, and a bootstrap 95% CI over eval items.
- Claims near the noise floor need at least 3 seeds per arm. Single-seed runs are fine for screening, but the claim is "screening result" until replicated.
- At small scale, prefer continuous metrics (log-likelihood of the correct answer, per-token loss) for ranking. Report accuracy too.

## 6. Three retrieval conditions, always

Every eval that involves facts is reported in all three conditions:

| Condition | What the model gets | What it measures |
|---|---|---|
| **closed-book** | question only | knowledge stored in weights |
| **oracle-context** | question + the gold supporting passage(s) | reasoning over given facts |
| **retrieved** | question + top-k from the retriever (or tool access) | the full open-book system |

The headline hypothesis lives in **oracle-context** and **retrieved**. **Closed-book** is the manipulation check: a knowledge-light model *should* lose closed-book score. If it doesn't, the data intervention failed, and no reasoning conclusion follows.

## 7. Compute accounting

Every run logs:

- parameter count, total and non-embedding
- tokens seen, and training FLOPs estimated as `6 × N_non-embedding × D`
- wall-clock hours, GPU model, number of GPUs
- cost, for rented GPUs

Comparisons across sizes plot against **non-embedding params** and **FLOPs**, not wall-clock.

## 8. Reproducibility

Every run records:

- git commit of the code (a dirty tree is not allowed for preregistered runs)
- the full resolved config
- the data manifest hash (which shards, which filter version)
- environment: Python, PyTorch and CUDA versions, GPU
- seeds

Checkpoints and logs stay out of git (see `.gitignore`). The experiment file links to where they live.

## 9. Hardware

- All runs are on rented vast.ai GPUs. No training or eval runs on the developer's local machine. Data and shards live on the Hugging Face Hub, and each run's data manifest names the Hub repo and revision.
- Within one comparison, keep all arms on the same GPU type when possible. If not, note it in **Deviations**.

## 10. Report honestly

- Negative and null results are written up with the same care as positive ones. "Curves stay parallel" is an answer, not a failure.
- If a run crashed, diverged or was stopped early, the experiment file says so with the numbers that exist.
- Never re-run a comparison with new seeds until it "works". If you run more seeds, report all of them.
