# Research Plan

This plan turns the README's question into phases. Each phase has experiments, metrics and a success criterion. Claims about prior work are sourced in [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md). The rules for running anything are in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## 0. Framing

**What we can and cannot show on one GPU.** A 150M model will not reason at frontier level. What we *can* measure at small scale is the **size ratio**: does a knowledge-light model reach a normal model's open-book score with fewer parameters? Does that ratio hold or grow as size increases? If the ratio is stable across 30M → 350M (and 760M+ on rented GPUs), that is evidence about the mechanism. Extrapolating it to frontier scale is a separate, flagged step.

**The two hypotheses.**

- **H1 (knowledge is dead weight for reasoning).** At a fixed open-book score, the knowledge-light recipe needs fewer non-embedding parameters. Its curve sits left of the normal curve.
- **H0 (knowledge is scaffolding).** Removing facts hurts reasoning even with retrieval. The curves are parallel, or knowledge-light sits to the right.

**The key measurement decomposition.** Every fact-involving eval is run in three conditions (see protocol §6):

```
closed-book     → how much the weights know       (should DROP for knowledge-light: manipulation check)
oracle-context  → reasoning over given facts      (H1 says: no drop, or better per param)
retrieved       → the full open-book system       (the headline number)
```

If knowledge-light doesn't lose closed-book score, the data intervention failed, and nothing about reasoning follows.

## 1. Defining "knowledge-light" data

This is the central design decision. Two complementary interventions:

### A. Entity renaming (primary, for the headline)

Replace **long-tail named entities** with consistent random surrogates **within each document** (e.g. `Marie Curie` → `Talven Orsk` everywhere in one document, a different surrogate in the next).

- Keeps: syntax, discourse, in-context coreference, causal and temporal structure, token count, domain mix.
- Removes: facts that build up *across* documents, which the model would otherwise memorize.
- Knob: the frequency threshold that defines "long-tail". Head entities (countries, major cities, common concepts) stay. Common sense and world structure depend on them. Sweeping the threshold *is* the Track 1 knee.
- Risks: surrogate names have odd tokenization statistics. Mitigate by sampling surrogates from a name-like distribution, and check that the BPE length distribution matches. Numbers and dates are facts too. Handle them as a second knob (perturb or leave), never silently.

Entity renaming keeps the data the same size and style. So a normal-vs-light comparison changes *only* knowledge content. This is why it's the headline intervention.

**Prior art.** KLLMs (Cohen et al. 2026, literature review §3) already pretrains on entity-anonymized text at 135M–3B. It uses typed placeholders (`PERSON184`), anonymizes all entities or a random fraction of them, and runs no RL. It reports a closed-book collapse plus gains on contextual QA, fact verification, hallucination detection and calibration, all at matched sizes. We build on that and do not claim the intervention as new. What we add:

1. **Size ratio:** the parameters needed for a matched score, not gains at a fixed size.
2. **Multi-step reasoning evals:** synthetic deduction, multi-hop, math. KLLMs reports none.
3. **Frequency-based anonymization:** remove long-tail entities, keep head concepts. This tests where the line between concept and fact falls.
4. **Realistic surrogate names vs typed placeholders:** a cheap ablation in Phase 1a.
5. **Post-training:** distillation and RL on a knowledge-light base (Phases 2–5).

Phase 0 checks whether KLLMs' code and models have been released, and reuses them if so.

### B. Document filtering (Track 1 ablations)

Score documents by **fact density** (named entities per token, numbers/dates per token, a reference-style classifier) and drop or downweight the most fact-dense slice. Refill to the same token budget from the remaining pool.

- More realistic as a "data recipe". But it changes style and domain mix at the same time, so it's a confounded comparison. It's used to map the data equilibrium, not for the headline.

### C. Slices for the equilibrium map

Fixed token budget, varied mixture of: narrative/fiction, dialogue, educational/textbook, code, math, instructions, encyclopedic, news. Measure what each slice contributes to language, common-sense and reasoning metrics. The question is: what is the smallest mix that keeps them?

## 2. Eval suite (frozen at the end of Phase 0 as `evals-v1`)

Chosen for **signal above noise at 30M–350M**. Final choice is made in Phase 0 by measuring the noise floor. A task whose seed std swamps the differences we care about gets dropped or switched to a continuous metric.

| Group | Purpose | Candidates |
|---|---|---|
| Language | must not degrade | val loss on a held-out *normal* web slice and on a held-out *renamed* slice; BLiMP; LAMBADA |
| Common sense | must not degrade | HellaSwag, PIQA, WinoGrande, ARC-Easy (scored by answer likelihood) |
| Knowledge (manipulation check) | should drop | closed-book TriviaQA / NQ-open subsets; LAMA-style cloze probes |
| Reasoning over given facts | the hypothesis | synthetic multi-hop over context (ProofWriter/CLUTRR-style, generated so no fact is memorizable); HotpotQA / MuSiQue with gold paragraphs |
| Open-book system | headline | the same QA with BM25 top-k over a fixed Wikipedia snapshot |

Notes:
- For base models, score multiple-choice by log-likelihood. Score generative QA by exact match / F1 *and* by likelihood of the gold answer.
- The synthetic reasoning set uses **fictional entities by construction**. It's the cleanest test of reasoning without knowledge, and it's immune to contamination.
- One retriever (BM25) and one corpus snapshot are fixed for all of Phases 0–2. Improving the retriever is out of scope until Phase 3.

## Phase 0 — Baseline, eval suite, noise floor

**Goal:** a training stack we trust, and numbers to judge every later result against.

1. **Reproduce GPT-2 small (124M)** on FineWeb, with a small llm.c/nanochat-style PyTorch stack. The reference is llm.c's run: HellaSwag 29.9 after 10B tokens (literature review §10). Match it within the noise floor at the same token count, or at a shorter budget if the noise floor shows that is enough to validate the stack.
2. **Measure throughput** on the local 4080 SUPER and on one rented 5090. All later token budgets are set from these measurements, not guessed.
3. **Build the eval harness** for all groups above, with closed-book / oracle-context / retrieved conditions.
4. **Noise floor:** 3 seeds of the baseline at the Phase 1 size and token budget. Record the std of every metric. Drop or replace metrics that are pure noise at this scale.
5. **Size ladder:** fix the model shapes (e.g. ~30M, 60M, 125M, 350M non-embedding; 760M+ rented) and the LR/batch rule per size. Use one small LR sweep at two sizes. The rule is then fixed, not re-tuned per arm.
6. Freeze `evals-v1`.

**Success:** baseline reproduces; every kept metric has a measured noise floor; the eval suite and size ladder are tagged and committed.

## Phase 1 — Data equilibrium (Track 1) and the headline curve

**1a. Renaming pilot (125M, fixed budget).** Normal vs entity-renamed at 3 long-tail thresholds, plus one arm with KLLMs-style typed placeholders at full anonymization. That arm is both a replication anchor and the surrogate-names-vs-tags ablation.
- Manipulation check: closed-book drops monotonically with threshold. The full-anonymization arm should collapse closed-book recall as KLLMs reports (LAMA falls from 34.7 to 3.9 at 1.7B). If it doesn't, the pipeline is broken. Stop and debug before going further.
- Watch: language and common-sense metrics. Where they leave the noise floor is the first estimate of the knee.

**1b. Slice ablations (125M, fixed budget).** Leave-one-slice-out and fact-density filtering (§1B, §1C). Output: a table of each slice's contribution to each metric group.

**1c. Headline curve.** Normal vs the chosen knowledge-light recipe across the size ladder, 3 seeds where budget allows, evaluated in all three conditions. Fit score vs log(non-embedding params) per recipe. Report the **param ratio** at matched open-book score, with a bootstrap CI.

**Success:** a curve pair with CIs, and a stated verdict for H1 vs H0 at base-model level. Either verdict counts.

**Expected snag:** base models at ≤350M may be near chance on open-book multi-hop QA. Then the synthetic reasoning set and likelihood metrics carry Phase 1. The real QA verdict waits for Phase 2 post-training, which is why Phase 2 re-runs the comparison.

## Phase 2 — Making it smart (Track 2)

**Question:** do distillation and RL from verifiable rewards (RLVR) lift a knowledge-light base as much as a normal base of the same size?

1. **Distillation (SFT):** reasoning traces from an open teacher model, on tasks where the supporting facts are *in context* (so the student is never trained to recall facts). Identical data for both bases.
2. **RLVR:** GRPO-style RL on tasks with checkable answers (math word problems with context, synthetic deduction, open-book QA with exact-match reward).
3. Re-run the Phase 1c comparison on post-trained models. The question is whether post-training widens, keeps or closes the gap.

**Success:** a post-trained normal-vs-light comparison at ≥2 sizes, with the same three conditions.

## Phase 3 — Knowing it doesn't know (Track 3)

**Question:** can the model reliably decide *answer* vs *look it up*, without hallucinating and without over-searching?

1. **Label known vs unknown per model:** sample the model closed-book several times. "Known" means consistently correct. This label is model-specific, and a knowledge-light model *should* have far more "unknowns".
2. **Train an action:** emit a `<search>` (or abstain) action. Start with SFT on the labels, then RL. Plain RL with sparse rewards is reported unstable at 0.5–1B (DGPO, literature review §7), so plan for teacher-guided RL from the start. Reward: correct answer +1, wrong answer strongly negative, each search a small cost.
3. **Metrics:**
   - hallucination rate (confident and wrong)
   - search rate
   - over-search rate (searched although closed-book would have been right)
   - under-search rate (answered and was wrong)
   - selective accuracy
   - end-to-end accuracy with retrieval
   - calibration (ECE) of a verbalized or logit confidence

**Success:** hallucination rate well below the untrained baseline at a stated over-search cost. Bonus: test whether knowledge-light models are *easier* to calibrate, because their boundary between known and unknown is sharper.

## Phase 4 — Self-directed lookup (Track 5)

**Question:** when it doesn't know, can the model write its own queries / code, verify, and answer?

1. Tools: search over the fixed corpus, a sandboxed Python interpreter, later a web search API.
2. Tasks need lookup + computation + verification (e.g. multi-hop questions with arithmetic over retrieved numbers). Answers are verifiable.
3. RL with outcome reward plus a step/tool-call cost, starting from the Phase 3 model.
4. Metrics: task success, tool calls per task, verification behaviour (does it check before answering?), failure modes. Log trajectories and inspect them by hand.

**Success:** a knowledge-light model with tools beats a same-size normal model without tools, and approaches a larger normal model with tools.

## Phase 5 — Compact chain-of-thought (Track 4)

**Question:** once CoT is reliable, can it be compressed into a short internal language without losing accuracy?

1. Start from the most reliable verbose-CoT model from Phases 2–4.
2. Compress with length-penalized RL / token-budget training, then try stepwise internalization or continuous latent thoughts.
3. Metrics: accuracy vs CoT tokens (a Pareto curve). For the synthetic tasks, a *faithfulness* check: can a decoder or checker expand the compact trace into a valid step-by-step proof?

**Success:** a point on the Pareto curve with ≥ 2× fewer reasoning tokens at accuracy within the noise floor of the verbose model.

## Risks and how we'd notice

| Risk | Signal | Response |
|---|---|---|
| Renaming breaks language, not just knowledge | language/common-sense metrics drop at every threshold | raise the threshold; use filtering as the main intervention; report it |
| Small models ignore context (Pandey 2026: models of 7B and below fail 85–100% on oracle context for questions they can't answer closed-book) | oracle-context ≈ closed-book for both recipes | treat context use as a trained skill: in-context-fact SFT in Phase 2 before judging H1/H0; report the base-model null honestly |
| Base models too small for any open-book signal | oracle-context ≈ chance at all sizes | lean on synthetic reasoning + likelihood metrics; move headline to post-trained models (Phase 2) |
| Small-scale rankings don't transfer | 1c ranking flips between 125M and 350M | report it; confirm the key comparison on a rented ≥760M run |
| Retriever quality dominates | retrieved ≪ oracle-context for both recipes | report oracle-context as the reasoning result; treat retrieval as a separate axis |
| Compute overrun | GPU-hours exceed spec cap | stop at cap, report partial results, re-plan |

## Decision points (owned by the user)

- After Phase 0: approve the eval suite, size ladder and rented-GPU budget.
- After Phase 1c: H1/H0 verdict. Decide whether Phases 2–5 use knowledge-light only, or keep both arms.
- Before any rented run over a stated cost.
