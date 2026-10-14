# 003 — Renamed-oracle diagnostic: name familiarity vs context use (eval only)

> Frozen sections above the line. See [`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md).

- **Status:** preregistered
- **Phase / track:** Phase 1a follow-up / Track 1. Eval only, no training.
- **Date preregistered:** 2026-10-14
- **Spec commit:** the commit that adds this file

## Question

In experiment 002, renamed models scored worse than normal models on PopQA *with the facts in context* (popqa_oracle). Is that because they **can't use context**, or because they **never saw real names** and handle them badly?

## Hypothesis

Name familiarity (`H_name`, from the 002 conclusion). The renamed arms were trained on text whose rare names are pseudo-words (r50/r80/r100) or typed tags (tag100). The PopQA oracle prompt is full of real names, which are out of distribution for them. Three pieces of evidence from 002 point this way:

- The context gain (oracle − closed gold_lp) rises with renaming.
- BLiMP, which is full of names, falls.
- Synth, which uses invented names, improves for surrogate arms.

The alternative (`H_ctx`) is that renaming damages reading from context itself.

Test: rename the names in the oracle items **the same way training renamed them**, with the training code (`rename_entities.rename_doc`). Then a renamed arm is on home turf, and the normal model faces unfamiliar names. Under `H_name`, the arms' deficit vs normal shrinks or flips. Under `H_ctx`, it persists.

## Prediction

Notation: Δ = arm − mean of the 3 normal seeds (e001 s1337/1338/1339), on gold log-prob per token, on the `dev` split. σ = the seed std of the normal models on that task. 002's real-name oracle Δ on dev: r50 −0.47, r80 −1.20, r100 −1.78, tag100 −1.92.

1. **`H_name`:** on its home-turf variant, the arm is **not worse** than normal: Δ ≥ −2σ.
   - Home turf is `popqa_oracle_surr` for r50/r80/r100 and `popqa_oracle_tag` for tag100.
   - I expect this for r100 and tag100, the arms whose 002 deficit was largest.
2. **Control:** on `popqa_closed_surr` (no context, renamed names), every model's acc is within ±0.03 of the majority baseline. With unknown pseudo-names there is nothing to recall, so this also checks for leakage.
3. Secondary, reported only: acc and acc_norm, the popularity buckets, and the `test` split.

## Decision rule

| Outcome (dev, home-turf variant, r100 and tag100) | Interpretation | Next |
|---|---|---|
| Δ ≥ −2σ for both | `H_name`: the 002 oracle drop was name unfamiliarity; renaming does not hurt reading | a knowledge-light recipe must keep real names in pretraining (real-name swap, or LMLM-style lookup); **user decision** on which |
| Δ ≤ ½ × the 002 real-name Δ for both, i.e. at least half the deficit remains | `H_ctx`: renaming hurts context use itself | flag against the headline hypothesis at this scale; prefer approaches that do not remove names (filtering, LMLM); **user decision** |
| anything in between, or r100 and tag100 disagree | both factors | report the size of each; **user decision** |
| control fails (closed_surr acc > majority + 0.03 for any model) | renaming leaks information, or the prior is exploitable | fix the eval before reading the main result |

## The one change

- **Varied factor:** the eval items. The PopQA oracle with real names vs the same items with names renamed by the training renamer.
- **Arms (models):** the 7 existing checkpoints: e001 s1337/1338/1339 (normal) and e002 r50/r80/r100/tag100. No retraining.
- **Held fixed:** checkpoints, PopQA items (same dedup, distractors, dev/test partition, fact order and templates), scoring code, GPU type (RTX 5090).

## Config

- **Eval suite** `evals-v0.3-dev` at this commit. Three tasks are new; the v0.2 tasks are unchanged.
  - `popqa_oracle_surr`: the oracle item renamed with arm `r100`, giving pseudo-word surrogates.
  - `popqa_oracle_tag`: the oracle item renamed with arm `tag100`, giving typed tags.
  - `popqa_closed_surr`: the closed-book question with the subject and entity answers renamed.
- **What is renamed:**
  - Every fact subject, i.e. all 4 subjects in the context, including the one in the question.
  - The 4 answer choices, when the relation's object is an entity: place of birth, father, mother, country, producer, director, screenwriter, composer, author, capital, capital of.
  - Common-noun objects are kept (occupation, genre, color, religion, sport), as training NER would also leave them.
  - Each item is one "doc", so a name gets the same replacement in the context, question and choices.
  - A leading "the" is not renamed, as with the training spans. The `Q:`/`A:` prompt scaffold is never renamed.
  - Typed-tag labels per relation are in `POPQA_ENT`.
- **Not mimicked:** training's top-100k-word blocklist for surrogates (`block` is empty here).
- **Leak check** (dev, local, CPU): items where an original name string still appears after renaming are 134/6532 (r100) and 8/6532 (tag100). Inspected cases are digit-only titles ("13", "2009"), which training also keeps, and one-letter titles ("A", "Q") that match the prompt scaffold, not real leaks.
- **Runs:** each checkpoint × {dev, test} × the 3 tasks. Outputs go to `evals/e003_<model>_<split>.json` in `lamshiu/efficient-lm-runs`.

## Budget

- **Compute:** one RTX 5090 instance (~$0.5–0.7/h). 14 eval jobs of 3 tasks × ~6.5k items. The v0.2 PopQA tasks took 7–15 s each on a normal host, so with setup this is ~20–40 min, **~$0.4**.
- **Hard cap:** `MAX_HOURS=1.5` (≤ $1.1), plus a local watchdog. The user approved this check on 2026-10-14 ("run b now, as it's checking not training").

---

*Everything below is filled in after the run. Do not edit anything above this line.*

## Deviations

## Results

## Conclusion
