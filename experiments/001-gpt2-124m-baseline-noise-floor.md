# 001 — GPT-2 124M baseline + noise floor (2.5B tokens, 3 seeds)

> Frozen sections above the line. See [`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md).

- **Status:** preregistered
- **Phase / track:** Phase 0 (baseline, eval check, noise floor)
- **Date preregistered:** 2026-10-13
- **Spec commit:** the commit that adds this file

## Question

Does our training stack produce a sane GPT-2 124M on FineWeb, and how much do the eval metrics vary from seed to seed at the budget Phase 1 will use?

## Hypothesis

- The stack is a faithful nanoGPT/llm.c-style GPT-2. llm.c's 124M reaches HellaSwag 29.9 after ~10B FineWeb tokens ([llm.c #481](https://github.com/karpathy/llm.c/discussions/481)).
- At 2.5B tokens (about 25% of that budget), the model should be clearly above chance but below the 10B reference.
- The eval harness reproduces published GPT-2 (HF `gpt2`) numbers. This is checked in the same job.

## Prediction

Per seed, 124M at 2.50B tokens:

- **Final val loss** (FineWeb val shard, 10.5M tokens): in [3.25, 3.55].
- **HellaSwag acc_norm** on the full 10,042-item validation set (our `test` split, the same set llm.c reports): in [0.265, 0.300].
- **Seed std** of HellaSwag acc_norm ≤ 0.006, and of final val loss ≤ 0.01.
- **HF `gpt2` reference on our harness:** HellaSwag acc_norm within ±0.01 of llm.c's 29.4 for the original GPT-2.

## Decision rule

| Outcome | Interpretation | Next |
|---|---|---|
| All 3 seeds in both predicted ranges, and HF `gpt2` check passes | stack and harness OK | adopt as the Phase 1 baseline config; noise floor = measured seed std per metric |
| Any seed diverges (loss spikes it never recovers from, NaN) or HellaSwag < 0.265 | stack bug or bad hyperparameters | debug before Phase 1; no Phase 1 runs |
| HF `gpt2` HellaSwag off by > 0.01 | harness scoring differs from llm.c | fix the harness; re-score saved checkpoints (no retraining) |
| A metric's seed std is ≥ the effect sizes Phase 1 needs to detect (> 1 point acc) | metric too noisy at this scale | rank Phase 1 on its continuous variant (gold log-prob) or drop it from `evals-v1` |
| Val loss in range but HellaSwag above 0.300 | stack fine; prediction too pessimistic | note it, adopt |

## The one change

- **Varied factor:** seed (1337, 1338, 1339). That is the measurement.
- **Arms:** 3 seeds of one config.
- **Held fixed:**
  - config `configs/gpt2_124m_2p5B.json`
  - data: FineWeb sample-10BT shards, HF `lamshiu/efficient-lm-fineweb10B-gpt2` commit `3702bbc2`; val shard + train shards 1–25
  - tokenizer: tiktoken gpt2
  - GPU type: RTX 5090, one per seed
  - eval suite: `evals-v0-dev` at this commit

## Config

- **Model:** GPT-2 small. 12 layers, d_model 768, 12 heads, context 1024, vocab 50304 (padded). 124.5M total params, 85.0M non-embedding.
- **Data:** FineWeb sample-10BT (HF `HuggingFaceFW/fineweb`, revision `9bb295dd`), GPT-2 tokens, llm.c shard format. Data manifest is `manifest.json` in the HF dataset repo. Train shards 1–25 (2.5B tokens) are read once in order, with no repetition.
- **Tokens:** 4768 steps × 524,288 = 2.50B.
- **Optimizer:** AdamW (β 0.9/0.95, eps 1e-8), weight decay 0.1 on 2D params, grad clip 1.0.
- **LR schedule:** peak lr 6e-4, 177 warmup steps (llm.c's 700/18865 scaled in proportion), cosine decay to 0.
- **Precision and compile:** bf16 autocast, `torch.compile`, micro-batch 16 with 32 accumulation steps.
- **Eval:** val loss every 250 steps. Final checkpoint scored with `evals.run` on all tasks (HellaSwag, PIQA, WinoGrande, ARC-Easy, LAMBADA, BLiMP), on both `dev` and `test` splits. `test` numbers are reported but not used for any choice in this experiment. HF `gpt2` is scored once on the `test` split, as the harness check, on the seed-1337 instance after training. Evals run after training, so an eval bug can't cost a training run; checkpoints are on HF and can be re-scored.

## Budget

- **Compute:** 6·N·D = 6 × 85.0M × 2.50B ≈ 1.3e18 FLOPs per seed.
- **Throughput:** measured on the 5090 at 190k tok/s (compile, micro-batch 16; 2026-10-13 benchmark log in HF `lamshiu/efficient-lm-runs/jobs/`). That gives ≈ 3.65 h training + ~0.3 h data/compile/eval ≈ **~4 h per seed**.
- **Cost:** about $0.50–0.60/h → ~$2.2 per seed, **~$6.5 for all three**.
- **Hard cap:** `MAX_HOURS=4.5` per instance (5 for seed 1337, which also runs the HF `gpt2` check), enforced on the instance (self-destroy) and by a local watchdog. Worst case 14 h × $0.60 ≈ $8.4, inside the remaining Phase 0 cap ($9.49 as of 2026-10-13).

---

*Everything below is filled in after the run. Do not edit anything above this line.*

## Deviations

## Results

## Conclusion
