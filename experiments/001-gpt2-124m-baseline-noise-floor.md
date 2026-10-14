# 001 — GPT-2 124M baseline + noise floor (2.5B tokens, 3 seeds)

> Frozen sections above the line. See [`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md).

- **Status:** done
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

- **Seed 1339 restarted on a new host.** The first instance (55860067, machine 145858) trained at ~38k tok/s while reporting 99% GPU. It was destroyed after ~1.5 h (~$1 wasted, no checkpoint kept). Seed 1339 was rerun from step 0, with the same config and seed, on instance 55871612 at ~171k tok/s. The other seeds ran at 190k (s1337) and 188k (s1338).
- **On-instance caps extended mid-run.** The `MAX_HOURS` watchdog was replaced over SSH on the training instances so that the queued post-training evals could finish. Local watchdogs stayed armed.
- **Phase 0 spend ~$10.3 against the $10 cap.** The overrun comes mainly from the wasted seed-1339 host. The user accepted it ("if it's not too much overbudge, just let it finish first").
- Nothing else changed. Config, data, eval suite and seeds are as frozen. Every `test` number below is reported only; none was used for a choice.

## Results

Final checkpoint (step 4768, 2.50B tokens). `ref` = HF `gpt2` (124M, trained by OpenAI on WebText) scored on our harness. Reproduce with `python scripts/summarize.py e001_gpt2_124m_2p5B_s1337 e001_gpt2_124m_2p5B_s1338 e001_gpt2_124m_2p5B_s1339 --split test --ref gpt2_test`.

**`test` split**

| metric | s1337 | s1338 | s1339 | mean | std | HF gpt2 |
|---|---|---|---|---|---|---|
| val_loss (FineWeb val) | 3.5557 | 3.5644 | 3.5668 | 3.5623 | 0.0058 | – |
| hellaswag.acc_norm | 0.2704 | 0.2725 | 0.2693 | 0.2707 | 0.0016 | 0.2955 |
| piqa.acc_norm | 0.6034 | 0.6045 | 0.6192 | 0.6090 | 0.0088 | 0.6181 |
| winogrande.acc | 0.5138 | 0.5107 | 0.5138 | 0.5128 | 0.0018 | 0.5162 |
| arc_easy.acc | 0.3897 | 0.3927 | 0.3822 | 0.3882 | 0.0054 | 0.4381 |
| arc_easy.acc_norm | 0.3577 | 0.3573 | 0.3413 | 0.3521 | 0.0094 | 0.3817 |
| blimp.acc | 0.7892 | 0.7827 | 0.7833 | 0.7851 | 0.0036 | 0.8250 |
| lambada.acc | 0.2164 | 0.2251 | 0.2179 | 0.2198 | 0.0047 | 0.3256 |
| lambada.ppl | 142.35 | 135.68 | 143.09 | 140.37 | 4.08 | 40.06 |
| hellaswag.gold_lp | -3.3256 | -3.3269 | -3.3503 | -3.3342 | 0.0139 | -3.0347 |
| piqa.gold_lp | -4.1212 | -4.1340 | -4.1670 | -4.1408 | 0.0236 | -3.8780 |

**`dev` split** (the one used for decisions; std is the noise floor)

| metric | mean | std |
|---|---|---|
| hellaswag.acc_norm | 0.2765 | 0.0017 |
| piqa.acc_norm | 0.6147 | 0.0036 |
| winogrande.acc | 0.5145 | 0.0033 |
| arc_easy.acc | 0.3825 | 0.0051 |
| arc_easy.acc_norm | 0.3365 | 0.0082 |
| blimp.acc | 0.7863 | 0.0018 |
| lambada.acc | 0.1654 | 0.0025 |

**Prediction check**

| prediction | result | |
|---|---|---|
| val loss in [3.25, 3.55] | 3.556 / 3.564 / 3.567 | **miss**, by 0.006–0.017 on every seed |
| HellaSwag acc_norm in [0.265, 0.300] | 0.270 / 0.273 / 0.269 | hit |
| seed std HellaSwag ≤ 0.006, val loss ≤ 0.01 | 0.0016, 0.0058 | hit |
| HF gpt2 HellaSwag within ±0.01 of 29.4 | 0.2955 | hit (+0.0015) |

No seed diverged: no NaN, and every run reached step 4768. Checkpoints, logs and eval JSONs are on HF `lamshiu/efficient-lm-runs` under `runs/e001_*` and `evals/e001_*`.

## Conclusion

The stack and the harness work. Adopt `configs/gpt2_124m_2p5B.json` as the Phase 1 baseline, with the seed std above as the noise floor.

- **The val-loss miss is narrow and does not look like a bug.** Every seed is within 0.017 of the bound. HellaSwag is inside its range, the HF `gpt2` harness check matches llm.c, and the seeds agree tightly. The predicted range came from llm.c curves measured on a different val set, so the absolute loss was not directly comparable. The prediction was miscalibrated; it is reported as a miss.
- **Noise floor.** On `dev`, no accuracy metric has seed std ≥ 1 point. ARC-Easy acc_norm (0.008) and PIQA (0.004–0.009 on test) are the noisiest. HellaSwag, WinoGrande, BLiMP and LAMBADA are ≤ 0.005. A std from 3 seeds is itself rough, so effects near 2σ should be read with care.
- **Not knowledge-light at this budget.** At 2.5B tokens the model is clearly below GPT-2 on LAMBADA and ARC-E, the most knowledge-heavy tasks. Phase 1 compares against this matched baseline, not against GPT-2.
- **Next:** experiment 002 (renaming pilot) uses seed 1337 as its normal arm. All three checkpoints are re-scored on `evals-v0.2-dev` there for the extended noise floor.
