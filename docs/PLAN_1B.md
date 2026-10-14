# Plan: the 1B race (normal vs knowledge-light, both taught to search)

Written 2026-10-14, after experiments 001–003. **Status: draft for the user's approval. No spending on it yet.**

## The claim we test

> A small model that never memorized rare trivia, but is trained to read and to say "I don't know → search", is at least as good as a normal model of the same size trained the same way. It also hallucinates less.

Two models, identical except for the pretraining text:

| | **A: normal** | **B: knowledge-light** |
|---|---|---|
| pretraining text | FineWeb as is | FineWeb, rare names swapped with other real names |
| size, tokens, settings | same | same |
| post-training ("I don't know → search") | same recipe | same recipe |

**B ≥ A** on reasoning-with-search, with fewer confident wrong answers → the cognitive-core idea holds at 1B. **B < A** → rare-fact knowledge helps reasoning (H0), and we measure by how much.

## Why this recipe (evidence, not our own pilot)

The user chose to skip a 124M swap pilot and rely on published evidence, plus a free early check inside the 1B run (§6).

1. **Changing meanings force reading.** Models learn to use context instead of memorizing when the data has many rare items whose meaning changes. A Zipfian mix of fixed common items and changing rare ones gives both abilities ([Chan et al., NeurIPS 2022](https://arxiv.org/abs/2205.05055)). Swapping rare names, while keeping famous ones, builds exactly that mix into real text. Caveat: their evidence comes from controlled toy tasks, not LM pretraining with name swaps.
2. **Names must stay real.** In our own work, made-up names hurt reading of real-name text. Experiment 003 showed that this was name unfamiliarity, not lost reading ability. Swapping in real names keeps them familiar.
3. **Reading exercises multiply data value.** Turning passages into passage + questions improves what a model learns from the same text ([Cheng et al., ICLR 2024](https://arxiv.org/abs/2309.09530); [Instruction Pre-Training, 2024](https://arxiv.org/abs/2406.14491)).
4. **Small models must be taught to use context.** Models under 7B often ignore retrieved text ([Pandey et al. 2026](https://arxiv.org/abs/2603.11513)). Explicit faithfulness training helps ([RHIO, ACL 2025](https://aclanthology.org/2025.acl-long.826)).
5. **Teachers save data.** Distilling from a stronger model was one of the approaches that worked in BabyLM ([findings](https://arxiv.org/abs/2504.08165)). For search, teacher guidance is needed at ≤1B, where plain RL is unstable ([DGPO](https://arxiv.org/abs/2508.20324)).

## 1. Model

- **Shape:** GPT-style decoder with 18 layers, d_model 2048, 16 heads (head dim 128), and GPT-2 BPE (vocab 50304, tied embeddings).
  - Parameters: 906M non-embedding, 1.01B total.
- **Positions:** RoPE replaces the learned position table (`wpe`), so the context is not capped by a table. **Context 4096** (the user's choice; can be extended later).
- **Unchanged:** the AdamW recipe, bf16, `torch.compile`, and fused attention (`F.scaled_dot_product_attention`).

## 2. Pretraining data

- **Source:** our FineWeb sample-10BT shards, 10.35B tokens (`lamshiu/efficient-lm-fineweb10B-gpt2`). One pass with no repetition. **10B tokens per model**, ~10 tokens/param (half of "Chinchilla"). This is a cost choice; see the budget for 5B and 20B.
- **A:** shards as is.
- **B:** the same docs with rare names swapped.
  - Run the existing NER pipeline over all 10B tokens (`data/rename_entities.py ner`).
  - Add a new `swap` arm: each rare surface (document frequency ≤ threshold) is replaced by a **real** entity surface of the same type (PERSON → PERSON, …), drawn from the corpus's own rare-entity pool.
  - Words map position by position, so partial mentions stay consistent ("Lovelace" → "Delgado"). Replacements are consistent within a doc and independent across docs. The rng is seeded per doc.
  - **Threshold:** the r50 rule (rename the rarest names covering ~50% of mentions; df ≤ 128 at 2.8B tokens). It is rescaled to the 10B corpus by the same mention fraction. Famous names stay as they are.
  - **Known wrinkles,** accepted and recorded: false statements are created by design; first-name gender can mismatch pronouns; spaCy-small NER errors carry over.
- **Reading-exercise slice:** *not* in v1 pretraining, to keep A vs B a single change. Reading practice enters in post-training (§4), identically for both.

## 3. Code changes (local editing only; CPU self-checks)

1. RoPE in `src/model.py`, with a round-trip and equivalence self-check.
2. **Multi-GPU (DDP)** in `src/train.py`, via `torchrun`. On one 5090, 10B tokens would take ~5–6 days. On 4× 5090, ~1.5 days.
3. **Resume from checkpoint.** Multi-day rented runs can die. Save a checkpoint every ~2 h to HF and resume from the latest.
4. `swap` arm in `data/rename_entities.py`, with a self-test.
5. A 1B benchmark job: tok/s and memory at 4K on 1× and 4× 5090. The budget below waits on its numbers.

## 4. Post-training: the "I don't know → search" dataset (a must)

Applied identically to A and B.

- **Search environment:** an **offline** Wikipedia index with BM25 (no GPU, free, never changes). Live web search is a later step.
- **Questions with checkable answers:**
  - single-fact: PopQA, Natural Questions, TriviaQA;
  - multi-step: HotpotQA, 2WikiMultihopQA, MuSiQue.
  - Licenses are to be checked per dataset. Fixed held-out splits for evaluation.
- **Knows or not, per model** (R-Tuning style, [repo](https://github.com/shizhediao/r-tuning)): ask each model closed-book several times. Reliably right → the example is "answer directly". Otherwise → "I don't know → search". Each model gets labels from its own knowledge.
- **Trajectories:** a teacher model writes short traces of 1–3 searches, under 4K tokens. They are kept only if the final answer is correct. Format:
  `<think>…I don't know X…</think> <search>q</search> <result>…</result> <think>…</think> <answer>…</answer>`
  Also include "searched, not found → say so". Target: ~30–50K trajectories.
- **Reading practice:** passage + questions made from Wikipedia passages, including multi-hop. This builds "answer from the text" before search is added.
- **Training order:**
  1. SFT on reading + trajectories.
  2. Teacher-guided RL with reward: correct +1, wrong −2 (more than abstaining), "I don't know" 0, each search −0.05.
- **Teacher choice (user decision, priced before use):** an open model on a rented GPU, or Claude via API.

## 5. Evaluation (held-out)

- **Language:** the existing suite (HellaSwag, PIQA, WinoGrande, ARC-E, BLiMP, LAMBADA).
- **Knowledge removal:** PopQA closed-book. B should drop.
- **Reading:** PopQA oracle with real names, HotpotQA with gold passages, synthetic multi-hop.
- **The race:** end-to-end accuracy with search; **confident-wrong rate** (hallucination); "I don't know" rate; over- and under-search; searches per question.

## 6. Go / no-go checkpoints

1. **After the benchmark:** if measured cost for A+B is > 1.5× the estimate, re-plan tokens or size before training.
2. **Early check at ~1B tokens (10%):** evaluate A and B. **Pause B and report** if either holds:
   - B is worse than A on the real-name PopQA oracle by > 0.3 nats/token (r50 with made-up names was −0.46);
   - B is worse on BLiMP by > 0.01.

   This replaces the skipped pilot.
3. **End of pretraining:** B's closed-book PopQA must drop. If not, the swap failed to remove knowledge, so stop before post-training.

## 7. Budget (estimates; the benchmark replaces them with measured numbers)

Throughput estimate: our 124M ran at ~1.5e14 effective FLOP/s on a 5090. At 1B / 4K, ~7e9 FLOP/token gives **~21K tok/s per 5090**. 5090s rented at $0.47–0.70/h on 2026-10-14.

| item | estimate |
|---|---|
| 1B benchmark (1× and 4×) | ~$2 |
| NER + swap over 10B tokens (256-core CPU box) | ~$5 |
| Pretraining A, 10B tokens (~130 GPU-h) | ~$80 |
| Pretraining B, 10B tokens | ~$80 |
| Evals (early check + final) | ~$5 |
| Teacher trajectories | TBD (priced per option before use) |
| Post-training SFT + RL, A and B | ~$20–60 (TBD after the data size is fixed) |
| **Total without teacher** | **~$190–230** |

Token-budget alternatives for pretraining A+B: **5B → ~$80**, **10B → ~$160**, **20B → ~$320**. 20B needs ~10B more FineWeb tokens prepared, which is cheap.

Current vast credit: $17.41, enough for the benchmark and the data stage only.

## Decisions needed from the user

1. Approve the plan, or change it.
2. Pretraining tokens: 5B / **10B (recommended)** / 20B.
3. Budget top-up for pretraining.
4. Later: teacher model choice, once priced.

## Order of work

1. Code changes (§3): free, starts on approval.
2. Benchmark (~$2) and data stage (~$5): within current credit.
3. Spec for experiment 004 (pretraining A vs B), committed before launch.
4. Pretraining with the early check.
5. In parallel: build the search dataset.
6. Post-training, then the race.
