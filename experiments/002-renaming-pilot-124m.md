# 002 — Entity-renaming pilot at 124M (Phase 1a)

> Frozen sections above the line. See [`docs/EXPERIMENT_PROTOCOL.md`](../docs/EXPERIMENT_PROTOCOL.md).

- **Status:** running
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

### Training

| arm | instance | started (ET) | tok/s |
|---|---|---|---|
| r50 | 55906528 (BE) | 06:02 | 188k |
| r80 | 55907606 (SE) | 06:11 | 173k |
| r100 | 55909069 (PL) | 06:21 | 190k |
| tag100 | 55909606 (VN) | 06:28 | 153k |

## Conclusion
