# 003 — Renamed-oracle diagnostic: name familiarity vs context use (eval only)

> Frozen sections above the line. See [`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md).

- **Status:** done
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

- **The `popqa_oracle_tag` variant has an index shortcut, found after the run** while inspecting items. Tags are numbered by first appearance, as in training, so the object tag's number often reveals the answer. Example: "The screenwriter of WORK_OF_ART_2 was PERSON_2 … Q: Who was the screenwriter for WORK_OF_ART_2?". For same-type relations the gold is the next number. Accuracy and gold_lp on this variant therefore measure index matching as well as context use. **tag100's home-turf result is not reported as evidence.** The surrogate variant has no such shortcut.
- The control's preregistered wording, "within ±0.03 of the majority baseline", misses narrowly on dev (acc 0.250–0.255 vs majority 0.283, a gap of 0.028–0.033). The failure condition in the decision table (acc > majority + 0.03) is not met. For renamed items the relevant floor is uniform chance (0.25): the renamed answers no longer carry the identity the majority prior uses.
- Instance 55966902 (RTX 5090, CA) ran 13:45–13:59 ET, 14 eval jobs, ~$0.2. No other changes.

## Results

Gold log-prob per token on **dev**. Δ = arm − mean of the 3 normal seeds; σ = the normal seed std. Test agrees to within ±0.03 on every Δ. Reproduce with `python scripts/summarize.py e003_<model> --split dev` (normal: `e003_e001s1337` etc.; arms: `e003_r100` etc.).

| task | normal (σ) | r50 Δ | r80 Δ | r100 Δ | tag100 Δ |
|---|---|---|---|---|---|
| popqa_oracle, real names (002) | -3.51 (0.07) | -0.47 | -1.20 | -1.78 | -1.92 |
| **popqa_oracle_surr** (surrogate names) | -4.09 (0.06) | **+0.12** | **+0.04** | **+0.23** | -1.00 |
| popqa_oracle_tag (typed tags, *shortcut, see Deviations*) | -3.26 (0.27) | -0.62 | -0.86 | -0.38 | +0.87 |
| popqa_closed_surr (no context) | -9.75 (0.07) | +0.81 | +0.79 | +0.71 | -0.25 |

Accuracy (dev; majority 0.283, uniform chance 0.25):

| task | normal | r50 | r80 | r100 | tag100 |
|---|---|---|---|---|---|
| popqa_oracle (real names, 002) | 0.304 | 0.284 | 0.268 | 0.262 | 0.271 |
| popqa_oracle_surr | 0.244 | 0.242 | 0.248 | 0.241 | 0.247 |
| popqa_closed_surr | 0.252 | 0.251 | 0.255 | 0.252 | 0.253 |

Context gain on surrogate items (oracle_surr − closed_surr gold_lp): normal +5.67, r50 +4.98, r80 +4.91, r100 +5.19.

**Prediction check:**

1. **`H_name` (Δ ≥ −2σ on home turf).**
   - r100 on surrogates: +0.23 (2σ = 0.12) → **hit**. The −1.78 real-name deficit disappears and slightly reverses.
   - r50 (+0.12) and r80 (+0.04) also hit.
   - tag100: home-turf result invalid (shortcut). On surrogates, which are foreign to it, it is −1.00, which fits the familiarity story.
2. **Control:** no leakage. Closed-book accuracy on renamed items is at chance (0.25) for every model.

## Conclusion

**Under the preregistered rule, the 002 oracle drop is explained by name unfamiliarity, not by damaged context use.**

- When the names in the prompt look like the names a model was trained on, renamed models are not worse than normal ones. The deficit follows the *name style*, not the arm:
  - surrogate-trained arms lose with real names and win with surrogates;
  - tag100 loses with surrogates.
- The decision-table row is "`H_name` → a knowledge-light recipe must keep real names in pretraining (real-name swap or LMLM-style lookup); **user decision**".

**Important limit.** At 124M and 2.5B tokens, **no model reads facts from context in accuracy terms**, normal included.

- Surrogate oracle accuracy is at chance for all models.
- With real names, the oracle does not beat closed-book even for normal models (0.304 vs 0.318).
- The gold_lp differences mostly track *string familiarity*: renamed arms already score +0.7 on surrogate answers *without* context. Their context gain on surrogate items (+4.9 to +5.2) is somewhat *below* normal's (+5.7), though the lower base makes that comparison loose.

So this experiment shows that renaming did not *break* reading. It cannot show that renaming *helps* reading: there was almost no reading at this scale to begin with. That needs a larger and longer-trained model with real names, which is the user's proposed direction (1B, longer context, real names).
