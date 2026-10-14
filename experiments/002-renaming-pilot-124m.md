# 002 — Entity-renaming pilot at 124M (Phase 1a)

> Frozen sections above the line. See [`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md).

- **Status:** done (awaiting user decision on Phase 1c recipe)
- **Phase / track:** Phase 1a / Track 1
- **Date preregistered:** 2026-10-14
- **Spec commit:** the commit that adds this file

## Question

At GPT-2 124M and 2.5B tokens, does renaming rare named entities in the pretraining text:

1. remove closed-book factual knowledge (the manipulation check), and
2. leave language, common sense, and reasoning over given facts intact?

Which renaming level (50 / 80 / 100% of entity mentions, rarest first) and which form (surrogate names vs typed tags) gives the best trade-off?

## Hypothesis

From the plan (§1A) and KLLMs (literature review §3):

- Rare entities carry most of the memorizable facts. Common entities and concepts carry common sense.
- Renaming should cut closed-book recall, most strongly for rare (tail) facts and more as coverage rises.
- Language and common sense should be roughly preserved at r50, with small losses expected at r100 and tag100.
- Reading facts from context (oracle) and in-context reasoning (synthetic) should not get worse, and may improve, as KLLMs found for contextual tasks.
- Surrogate names (r100) should beat typed tags (tag100) on language metrics, because the text stays natural.

## Prediction

All four arms are compared with the normal arm on the `test` split. Noise floor σ = the seed std from experiment 001 (3 seeds), for each metric. "Drop" means worse by > 2σ.

1. **Manipulation check.** popqa_closed gold log-prob/token drops for r100 and tag100. The drop is ordered r50 ≤ r80 ≤ r100. In r50 the tail bucket drops more than the head bucket.
2. **Language and common sense.** For r50, there is no drop on HellaSwag acc_norm, PIQA acc_norm, WinoGrande acc or BLiMP acc. For r100 and tag100, each of these drops by no more than 0.01 absolute. ARC-Easy and LAMBADA may drop, because both partly test knowledge.
3. **Context and reasoning.** For every arm, popqa_oracle gold log-prob/token and synth_1hop acc show no drop.
4. **Form.** r100 is at least as good as tag100 on each language and common-sense metric.
5. **Val loss.** Loss on the normal FineWeb val shard is higher for every renamed arm than for normal: the arms never saw real rare names. Each arm's loss on its own renamed val shard is reported too.

## Decision rule

| Outcome | Interpretation | Next |
|---|---|---|
| popqa_closed shows no drop for r100 and tag100 | intervention failed, or the 124M base knows too little to remove | **stop**; check closed-book signal on the normal 001 models vs chance; consider a larger model or a different probe before Phase 1c |
| Knowledge drops and some arm keeps language and common sense within the noise floor | a usable knowledge-light recipe exists | pick the most aggressive such arm as the candidate for the Phase 1c headline curve (**user decision**) |
| Knowledge drops, but every arm loses language or common sense beyond the noise floor | knowledge removal costs language at 124M | report the size of the trade-off, then ask the user: accept the cost, or move to filtering (§1B) |
| tag100 beats r100 on language metrics | surrogate names hurt more than tags | prefer tags in Phase 1c |
| popqa_oracle or synth gets worse in renamed arms | removing knowledge hurts context use or reasoning (supports H0) | flag prominently; it bears on the headline hypothesis |

## The one change

- **Varied factor:** the pretraining text: normal vs entity-renamed at a given coverage and form.
- **Arms:**
  - **normal:** experiment 001 seed 1337, reused rather than retrained. Same config and seed. Its training docs differ slightly from the renamed arms: renamed train starts at the first doc after the val docs.
  - **r50, r80, r100:** surrogate names, rarest entities first, until 50 / 80 / 100% of entity mentions are covered.
  - **tag100:** typed tags (`PERSON_1` …) on 100% of entity mentions.
- **Held fixed:**
  - `configs/gpt2_124m_2p5B.json`, seed 1337, 2.50B tokens (4768 steps)
  - tokenizer (GPT-2 BPE, no new tokens)
  - GPU type: RTX 5090
  - eval suite `evals-v0.2-dev` at this commit

## Config

- **Data.** `lamshiu/efficient-lm-fineweb-renamed` `<arm>/`, built by `data/rename_entities.py` at commit 71052aa from the same FineWeb sample-10BT docs:
  - spaCy `en_core_web_sm` 3.8.0 NER over 11 entity types; numbers and dates are kept.
  - Threshold = document frequency of the surface string.
  - Repeats of a renamed string within a doc are also renamed (case-sensitive, whole word).
  - Surrogates are consistent within a doc, independent across docs, and never a top-100k common word or a known entity word.
  - Achieved coverage per arm (`achieved_mention_frac`, threshold, token ratio, tokens changed) is recorded in each arm's `manifest.json` on HF and copied into Results. It doesn't change any prediction.
- **Model and optimizer:** identical to experiment 001.
- **Evals** (each final checkpoint, `dev` and `test`):
  - the six base tasks, plus popqa_closed and popqa_oracle (with popularity buckets and majority baseline), synth_1hop and synth_2hop
  - `evals.val_loss` on the normal val shard and on the arm's own val shard
- **Noise floor:** the 3 checkpoints from experiment 001, re-scored on `evals-v0.2-dev` in the same eval job.

## Budget

- **Data stage:** one 256-core instance, about 2.5 h at $0.68/h (NER measured at ~700 docs/s on 245 workers), so about $1.7.
- **Training:** 4 arms on 4 × RTX 5090, about 4 h at about $0.6/h, so about $2.4 per arm and about $9.6 in total.
- **Evals:** one eval-only instance, about 0.5 h, about $0.3.
- **Total:** about $11.6, against the $15 the user approved for Phase 1a overnight (2026-10-14).
- **Hard caps:** `MAX_HOURS` 4.5 per training instance and 4 for the data stage, plus local watchdogs. If the projected spend exceeds $15, stop launching and report.

---

*Everything below is filled in after the run. Do not edit anything above this line.*

## Deviations

- **Training commit is `cb8c6bf`, not the spec commit.** The diff since `9b97f2b` touches only `experiments/001` and `README.md`, so code, configs and evals are identical.
- **Run names:** `e002_gpt2_124m_2p5B_<arm>`, seed 1337. Training only on each instance; all evals run later on one instance, as planned.
- **tag100 cap extended.** Its host runs at 153k tok/s, so training needs ~4.6 h, more than `MAX_HOURS=4.5`. At 06:39 ET the on-instance watchdog was replaced over SSH (fires ~11:23 ET, ~4.9 h total, ~$3.1) and the local watchdog was moved to ~11:40 ET. Extending was preferred to re-renting: it costs ~$0.5 more but avoids the risk of another slow or stuck host.
- **Evals split across two instances.** The 3 normal checkpoints (e001 s1337/1338/1339) were scored early, on instance 55911613 (06:41–06:56 ET, ~$0.16), while the arms trained. That checked the new v0.2 tasks at full size before they mattered. The arms are scored later on a second instance. Both use the same commit `cb8c6bf`, the same GPU type and the same suite. Outputs are `evals/e002_e001s<seed>_{dev,test,valnormal}.json`.
- `scripts/summarize.py` now also prints the PopQA popularity-bucket metrics. This is reporting only; scoring is unchanged.
- **Data stage** (instance 55892933) finished at ~06:24 ET, 2026-10-14, then self-destroyed. NER took ~95 min at ~716 docs/s on 245 workers; the 4 arms took ~8 min each.

## Results

### Data (from each arm's `manifest.json` on HF)

NER pass: 4.0M docs (143,468 val + train), 98.9M entity mentions, 20.7M distinct (surface, label) keys, 17.2M of them seen in only one doc. Largest labels by mentions: ORG 35.5M, PERSON 31.3M, GPE 16.8M.

| arm | df threshold | mention frac | renamed / mentions | propagated | residual per renamed | tokens changed | token ratio | train tokens | train shards |
|---|---|---|---|---|---|---|---|---|---|
| r50 | 128 | 0.500 | 49,446,884 / 98,869,392 | 5,216,255 | 0.056 | 0.097 | 1.033 | 2.76B | 28 |
| r80 | 5,229 | 0.800 | 79,091,761 / 98,869,392 | 8,801,858 | 0.026 | 0.129 | 1.047 | 2.79B | 28 |
| r100 | 162,864 | 1.000 | 98,869,392 / 98,869,392 | 9,692,743 | 0.016 | 0.147 | 1.059 | 2.82B | 29 |
| tag100 | 162,864 | 1.000 | 98,869,392 / 98,869,392 | 9,692,743 | 0.014 | 0.141 | 1.044 | 2.78B | 28 |

Training reads the first 2.50B train tokens of each arm (4768 steps). Every arm has more than that, so no data repeats. Because the token ratio is above 1, each arm sees ~3–6% fewer documents than the normal arm.

### Noise floor on `evals-v0.2-dev` (normal arm, 3 seeds, `dev` split)

`evals.val_loss` on the normal val shard reproduces the training-time val loss (s1339: 3.5668 in both), a consistency check.

| metric | mean | std | chance / majority |
|---|---|---|---|
| popqa_closed acc | 0.318 | 0.001 | 0.283 |
| popqa_closed head acc | 0.394 | 0.004 | 0.261 |
| popqa_closed tail acc | 0.273 | 0.005 | 0.298 |
| popqa_closed gold_lp | -6.97 | 0.065 | – |
| popqa_closed tail gold_lp | -7.62 | 0.121 | – |
| popqa_oracle acc | 0.304 | 0.004 | 0.278 |
| popqa_oracle head acc | 0.354 | 0.006 | 0.261 |
| popqa_oracle gold_lp | -3.51 | 0.070 | – |
| synth_1hop acc | 0.295 | 0.002 | 0.25 |
| synth_1hop gold_lp | -1.65 | 0.091 | – |
| synth_2hop acc | 0.243 | 0.006 | 0.25 |
| synth_2hop gold_lp | -1.91 | 0.085 | – |

**A floor effect, noted before any arm is scored.** At 124M and 2.5B tokens, the normal model has closed-book signal only on head (popular) PopQA facts. On tail facts it is at the majority baseline. It barely uses the oracle context: oracle acc 0.304 vs majority 0.278, and on the head bucket oracle (0.354) is *below* closed-book (0.394). It is near chance on synth 1-hop and at chance on 2-hop. So for prediction 3, "no drop" is weakly informative in accuracy terms, and the gold log-prob variants carry most of the signal. For prediction 1, the tail-bucket accuracy is already at floor, so a tail-vs-head comparison must use gold_lp.

### Training

| arm | instance | started (ET) | tok/s |
|---|---|---|---|
| r50 | 55906528 (BE) | 06:02 | 188k |
| r80 | 55907606 (SE) | 06:11 | 173k |
| r100 | 55909069 (PL) | 06:21 | 190k |
| tag100 | 55909606 (VN) | 06:28 | 153k |

All 4 arms reached step 4768 with no divergence. Checkpoints: `runs/e002_gpt2_124m_2p5B_<arm>/ckpt_004768.pt`. Evals: `evals/e002_<arm>_{dev,test,valnormal,valown}.json`. One eval host was stuck in `loading` for 6 min and was destroyed and re-rented. The tag100 eval host ran 3–5× slower but finished.

### Arms vs normal (`test` split, the split the predictions are scored on)

Δ = arm − mean of the 3 normal seeds. z = Δ / seed std. The seed std comes from **3 seeds that share one data order, and it does not include item-sampling noise**, so z overstates significance for accuracy metrics. For example, WinoGrande's seed std is 0.0018, while its binomial SE is about 0.014. Gold log-prob (`gold_lp`, nats/token of the gold answer) is less noisy. Per-arm numbers: `python scripts/summarize.py e002_<arm> --split test`. Normal: `e002_e001s1337 e002_e001s1338 e002_e001s1339`.

| metric | normal | r50 | r80 | r100 | tag100 |
|---|---|---|---|---|---|
| val loss, normal FineWeb val | 3.562 | 3.650 | 3.736 | 3.787 | 3.782 |
| val loss, own arm val | – | 3.609 | 3.607 | 3.603 | 3.139 |
| hellaswag acc_norm | 0.2707 | 0.2686 | 0.2682 | 0.2688 | 0.2713 |
| piqa acc_norm | 0.6090 | 0.6094 | 0.6115 | 0.6077 | 0.6066 |
| winogrande acc | 0.5128 | 0.5351 | 0.5067 | 0.5233 | 0.5146 |
| blimp acc | 0.7851 | 0.7799 | 0.7721 | **0.7613** | **0.7623** |
| arc_easy acc | 0.3882 | 0.3855 | 0.3939 | 0.3792 | 0.3830 |
| lambada acc | 0.2198 | 0.2078 | 0.1628 | 0.1570 | 0.1300 |
| popqa_closed acc (majority 0.283) | 0.3148 | 0.2950 | 0.2729 | 0.2642 | 0.2680 |
| popqa_closed gold_lp | -7.00 | -7.93 | -8.96 | -9.50 | -9.18 |
| popqa_closed head gold_lp | -6.44 | -7.41 | -8.95 | -9.42 | -9.11 |
| popqa_closed tail gold_lp | -7.67 | -8.48 | -9.09 | -9.88 | -9.47 |
| popqa_oracle acc (majority 0.278) | 0.2980 | 0.2862 | 0.2723 | 0.2671 | 0.2639 |
| popqa_oracle gold_lp | -3.60 | -4.06 | -4.80 | -5.35 | -5.49 |
| context gain: oracle − closed gold_lp | +3.40 | +3.87 | +4.16 | +4.15 | +3.68 |
| synth_1hop acc (chance 0.25) | 0.276 | 0.286 | 0.260 | 0.271 | 0.275 |
| synth_1hop gold_lp | -1.69 | -1.36 | -1.45 | -1.42 | -2.09 |
| synth_2hop acc (chance 0.25) | 0.244 | 0.261 | 0.252 | 0.258 | 0.277 |
| synth_2hop gold_lp | -1.86 | -1.56 | -1.53 | -1.49 | -2.26 |

On **dev** the picture is the same, with one exception: r50 BLiMP drops 0.0075 (z −4.1), and synth_1hop acc falls 0.03–0.04 for the surrogate arms. Neither replicates on test.

### Prediction check (`test`)

| # | prediction | result |
|---|---|---|
| 1 | closed-book gold_lp drops for r100, tag100 | **hit**: −2.51, −2.18 nats/token |
| 1 | ordered r50 ≤ r80 ≤ r100 | **hit**: −0.93 / −1.96 / −2.51 |
| 1 | in r50, tail drops more than head | **miss**: head −0.97 vs tail −0.81 |
| 2 | r50: no drop on HellaSwag, PIQA, WinoGrande, BLiMP | **hit on test** (BLiMP z −1.4). On dev, BLiMP z −4.1 |
| 2 | r100, tag100: each drop ≤ 0.01 | **miss on BLiMP** (−0.024, −0.023). HellaSwag, PIQA and WinoGrande hit |
| 3 | popqa_oracle gold_lp: no drop, every arm | **miss**: drops −0.46 (r50) to −1.89 (tag100) |
| 3 | synth_1hop acc: no drop | **hit on test** (all near chance). On dev, surrogate arms −0.03 to −0.04 |
| 4 | r100 ≥ tag100 on each language metric | **mixed**, all within noise: r100 better on PIQA and WinoGrande, tag100 on HellaSwag and BLiMP |
| 5 | normal-val loss higher for every renamed arm | **hit**: +0.09 to +0.22 |

## Conclusion

**Renaming removes closed-book knowledge, and the effect grows with coverage. It also makes the model worse at reading real names supplied in context.** Under the decision rule:

- Closed-book drops, so this is not a "stop".
- On dev, no arm keeps every language metric inside the noise floor. r50 fails only on BLiMP (−0.0075), and that drop does not replicate on test. r100 and tag100 lose about 0.024 on BLiMP on both splits. → **report the trade-off and ask the user.**
- **Flag: popqa_oracle gets worse in every renamed arm**, roughly in proportion to coverage. This is the row that bears on the headline hypothesis.
- tag100 vs r100: no clear winner on language. On synthetic reasoning over fictional names, surrogates *help* (gold_lp +0.3 to +0.4) and tags *hurt* (−0.4).

**Interpretation: the evidence points to a name-familiarity confound more than to a reasoning deficit.** This is not proven. Five observations support it:

1. Oracle gold_lp falls with coverage even though the facts are given in context.
2. The *context gain* (oracle − closed gold_lp) **rises** with renaming: +3.40 normal vs +3.87 to +4.16 for the surrogate arms. Renamed models benefit more from context but start from a lower base on real names.
3. BLiMP, whose sentences are full of proper names, falls with coverage. The other language metrics do not.
4. On synth, which uses invented names, surrogate-trained arms gain while tag-trained arms lose. Models handle names the way their training text presented them.
5. The head bucket drops more than the tail, even at r50, which renames only entities with document frequency ≤ 128. The gold answers are real entity strings, so gold_lp also measures how readily the model writes real names.

**Limits:**

- 124M at 2.5B tokens is near the floor on context use. Normal models get oracle acc 0.298 vs majority 0.278, and synth is near chance. So accuracy-based claims about reasoning are weak here.
- Each arm is a single run. The normal arm's training docs differ slightly from the arms'.
- spaCy-small NER false positives are not measured.

**User decision needed (Phase 1c recipe).** Options:

- **A. Eval-only diagnostic (~$0.5, no training):** a renamed PopQA oracle that applies the same surrogate or tag renaming to the context, question and choices. It separates "can't use context" from "doesn't know real names". *Recommended first.*
- **B. New recipe, real-name swap:** replace rare entities with other real names of the same type, consistent within a doc. Names stay in-distribution while the facts break. Needs a new data run (~$1.5) plus training (~$2.4 per arm).
- **C. Adopt r50** as the knowledge-light recipe. It has the smallest language cost (within noise on test), but the closed-book drop is smaller and the oracle drop is −0.46.
- **D. Move to filtering** (§1B) instead of renaming.

**Spend:** about $13 for Phase 1a (credit $30.9 → $17.6), against the $15 cap. That covers the data stage, 4 training runs (one cap extended) and 5 eval instances.
