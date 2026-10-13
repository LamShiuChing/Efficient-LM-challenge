# Literature Review

What has been tried, what it showed, and the gaps this project targets. Every entry links to its primary source. Numbers are quoted only where they were read on that source. Coverage is current to **2026-10-13** and is not exhaustive. Add to it as you read.

## 1. How much of a model is knowledge?

- **Allen-Zhu & Li, *Physics of Language Models Part 3.3: Knowledge Capacity Scaling Laws* (2024)** — [arXiv:2404.05405](https://arxiv.org/abs/2404.05405). LMs store about **2 bits of knowledge per parameter**, even when quantized to int8. This is the quantitative basis for "parameters are spent on facts".
- **Allen-Zhu & Li, *Part 3.1: Knowledge Storage and Extraction* (2023)** — [arXiv:2309.14316](https://arxiv.org/abs/2309.14316). Memorized facts can be *extracted* reliably only if pretraining data is augmented (paraphrase, shuffling). Otherwise facts can be memorized and still be inaccessible.
- **Allen-Zhu & Li, *Part 3.2: Knowledge Manipulation* (2023)** — [arXiv:2309.14402](https://arxiv.org/abs/2309.14402). Models retrieve stored facts well but fail to classify or compare them without CoT, and they fail at inverse search.
- **Ruis et al., *Procedural Knowledge in Pretraining Drives Reasoning in LLMs* (ICLR 2025)** — [arXiv:2411.12580](https://arxiv.org/abs/2411.12580). Influence analysis: factual answers trace back to specific documents, while reasoning draws on *procedural* documents (formulae, code). This supports the idea that the two can be separated.

**Takeaway.** Facts and procedures appear to come from different data. A knowledge-light corpus should keep procedural text (math, code, worked explanations) and remove entity-bound facts.

## 2. The "cognitive core" idea

- **Karpathy on Dwarkesh Podcast, "AGI is still a decade away" (2025-10-17)** — [dwarkesh.com/p/andrej-karpathy](https://www.dwarkesh.com/p/andrej-karpathy). Karpathy describes a core "stripped from knowledge but contains the algorithms", suggests about 1B params may be enough, and says it should know what it doesn't know and look things up. An earlier tweet ([x.com/karpathy/status/1938626382248149433](https://x.com/karpathy/status/1938626382248149433), 2025-06-27) frames it as "a few billion param model that maximally sacrifices encyclopedic knowledge for capability". The tweet's wording was read via a secondary quote.
- **Savkin et al., *OCC-RAG: Optimal Cognitive Core for Faithful Question Answering* (2026-05)** — [arXiv:2606.00683](https://arxiv.org/abs/2606.00683). Released 0.6B and 1.7B models are trained on more than 3M synthetic multi-hop QA examples with citations. They match or beat general-purpose models 2–6× their size on HotpotQA, MuSiQue and related sets. This is task-specific SFT on existing bases, *not* knowledge-light pretraining.

## 3. Closest prior art: separating knowledge from the model

These papers are the nearest neighbours. The project must cite them and position against them.

- **Cohen et al., *Knowledgeless Language Models (KLLMs)* (2026-07-14)** — [arXiv:2607.12831](https://arxiv.org/abs/2607.12831). **This is essentially our §1A intervention.** The details below come from the full text, read via a summarizer that covered about 87% of the paper. Re-check any number before quoting it in a paper.
  - **Method.**
    - Flair OntoNotes NER tags people, NORP, facilities, organizations, GPEs, locations, products, events, works of art, laws and languages.
    - Each entity is replaced with a *typed placeholder* (e.g. `PERSON184`) that is consistent within a document. There is no coreference resolution.
    - Numbers, dates and quantities are kept. Estimated NER recall is about 87%.
    - The tokenizer is a 30k BPE trained on the anonymized text.
  - **Setup.**
    - SmolLM 135M/360M/1.7B and Llama-3.2 1B/3B shapes, trained from scratch on 2.5B tokens (Wikipedia + CNN/DM) or on 10B/20B tokens (smollm-corpus).
    - Each model has a matched twin trained on the original text. Only the anonymization differs.
    - Fine-tuned per benchmark.
  - **Results** (SmolLM-1.7B):
    - Closed-book LAMA falls from 34.7 to 3.9.
    - FEVER F1 rises from 86.9 to 94.7. HaluBench rises from 64.3 to 74.7.
    - With BM25 at 10B tokens: NQ rises from 35.7 to 44.9 and TriviaQA from 52.8 to 63.4. That is about 20–26% relative; FEVER gains only about 6%.
    - ECE falls from 0.115 to 0.083.
    - Unsupported answers in tool-augmented QA fall from 21.5% to 13.2%.
    - Under truncated or perturbed context, the anonymized model degrades less and abstains more.
    - Gains are present at 135M.
    - Random masking, span corruption and entity shuffling baselines give only marginal gains.
  - **Not covered:**
    - no math, no multi-hop eval; StrategyQA is the closest, 61.8 → 66.7
    - no matched-score comparison across sizes
    - no frequency-based (long-tail only) anonymization; ablations use a random fraction p ∈ {0.5, 0.75, 1.0}
    - no realistic surrogate names
    - no distillation or RL
    - at most 20B tokens
  - **Status.** Some numbers in the text don't match the tables (abstention F1, the 1.7B commonsense gains). Code and models are "to be released upon publication", and no links had appeared by 2026-10-13.
- **Zhao et al., *Limited Memory Language Models (LMLM)* (2025; ICLR 2026)** — [arXiv:2505.15962](https://arxiv.org/abs/2505.15962). Facts are stored in an external database during pretraining, and retrieved fact values are **masked from the loss**, so the model learns to look facts up instead of memorizing them. The models are 124M–382M and are reported competitive with much larger LLMs on factual precision.
- **Pouransari et al., *Pretraining with Hierarchical Memories: Separating Long-tail and Common Knowledge* (2025)** — [arXiv:2510.02375](https://arxiv.org/abs/2510.02375). A 160M "anchor" model plus an 18M memory fetched from a 4.6B memory bank matches a regular model with more than 2× the parameters.
- **Samuel et al., *More Room for Language: Investigating the Effect of Retrieval on Language Models* (2024)** — [arXiv:2404.10939](https://arxiv.org/abs/2404.10939). Retrieval-pretrained models store less world knowledge. They model local context better and global context worse. This is evidence that removing knowledge pressure reallocates capacity.
- **Memory layers:** Lample et al. 2019 ([arXiv:1907.05242](https://arxiv.org/abs/1907.05242)) and Berges et al. 2024, *Memory Layers at Scale* ([arXiv:2412.09764](https://arxiv.org/abs/2412.09764)). These move knowledge into sparse lookup parameters inside the model. It is an architectural alternative to removing it from the data.

**Gap.** KLLMs and LMLM show that knowledge-light pretraining helps *evidence-grounded factual* tasks. What is still open is whether it changes the **parameters needed for reasoning**. That means a score-vs-size curve on multi-step reasoning with facts supplied, compared against a matched normal recipe. It also means asking whether knowledge-light bases respond better to reasoning post-training. That is this project's headline.

## 4. Does reasoning need stored knowledge? (the counter-hypothesis)

- **Ye et al., *LIMO: Less is More for Reasoning* (2025)** — [arXiv:2502.03387](https://arxiv.org/abs/2502.03387). Reaches 63.3% AIME24 with about 1% of prior SFT data. The "LIMO Hypothesis": complex reasoning emerges from minimal demonstrations **when domain knowledge is already encoded in pretraining**.
- **Gu et al., *SynthWorlds* (2025)** — [arXiv:2510.24427](https://arxiv.org/abs/2510.24427). Parallel real and synthetic worlds with equal reasoning difficulty. Models do better in the real-mapped world, both closed-book and with RAG. Knowledge-integration methods shrink this "knowledge advantage gap" but do not remove it. **This is direct evidence for H0** in models pretrained normally.
- **Pandey et al., *Can Small Language Models Use What They Retrieve?* (2026-03)** — [arXiv:2603.11513](https://arxiv.org/abs/2603.11513). Five sizes from 360M to 8B, with no retrieval, BM25, dense and oracle retrieval. With *oracle* retrieval, models of 7B or below fail 85–100% of the time on questions they cannot answer closed-book. Adding context also loses 42–100% of answers that were previously right. The dominant failure is ignoring the context. **This is the biggest risk to the project.** At small scale, the bottleneck may be *using* context, not storing facts. It also motivates training context use explicitly (Phase 2) instead of assuming it.

## 5. Data recipes at small scale

- **Magnusson et al., *DataDecide* (2025)** — [arXiv:2504.11393](https://arxiv.org/abs/2504.11393). The ranking at a single small size (150M) is "a strong baseline for predicting best models at our larger target scale (1B) (~80% of comparisons correct)". This is pairwise decision accuracy over data recipes. It justifies making data decisions at ~150M.
- **Eldan & Li, *TinyStories* (2023)** — [arXiv:2305.07759](https://arxiv.org/abs/2305.07759). Models under 10M params produce fluent, consistent stories when the data is narrow and simple. Language competence is cheap when the data is designed for it.
- **Gunasekar et al., *Textbooks Are All You Need* (phi-1, 2023)** — [arXiv:2306.11644](https://arxiv.org/abs/2306.11644). 1.3B params trained on about 7B tokens of "textbook quality" data reach 50.6% HumanEval. Data quality can substitute for scale.
- **Penedo et al., *FineWeb / FineWeb-Edu* (2024)** — [arXiv:2406.17557](https://arxiv.org/abs/2406.17557). FineWeb is 15T tokens. FineWeb-Edu is a 1.3T educational subset that is much stronger on knowledge and reasoning benchmarks (MMLU, ARC). These are candidate source pools for Phase 0/1.
- **Li et al., *DataComp-LM* (2024)** — [arXiv:2406.11794](https://arxiv.org/abs/2406.11794). A controlled benchmark for data curation. Model-based filtering is the key ingredient.
- **SmolLM2** ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)) and **SmolLM3** ([HF blog, 2025-07-08](https://huggingface.co/blog/smollm3)) are fully documented small-model recipes. They include released datasets (FineMath, Stack-Edu, SmolTalk) that are useful as procedural slices.
- **Hoffmann et al., *Chinchilla* (2022)** — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556). Compute-optimal scaling, the reference for setting token budgets per size.

## 6. Making small models reason (Track 2)

- **DeepSeek-AI, *DeepSeek-R1* (2025)** — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) (v1 §4.1). Distilling a strong reasoner into smaller models beats running RL directly on them. R1-Distill-Qwen-32B scores 72.6 on AIME24, against 47.0 for RL-only Qwen-32B. The v2 abstract no longer states this, so cite v1.
- **Muennighoff et al., *s1: Simple test-time scaling* (2025)** — [arXiv:2501.19393](https://arxiv.org/abs/2501.19393). 1,000 curated examples plus "budget forcing" control how long the model thinks.
- **Ye et al., *Physics of LMs Part 2.1: Grade-School Math* (2024)** — [arXiv:2407.20311](https://arxiv.org/abs/2407.20311). A controlled synthetic study of how LMs actually solve GSM-style problems. It is the methodological model for our synthetic reasoning evals.

## 7. Retrieval and tool use (Tracks 3 and 5)

- **Borgeaud et al., *RETRO* (2021)** — [arXiv:2112.04426](https://arxiv.org/abs/2112.04426). With a 2T-token retrieval database, it is comparable to GPT-3 on the Pile with 25× fewer parameters.
- **Izacard et al., *Atlas* (2022)** — [arXiv:2208.03299](https://arxiv.org/abs/2208.03299). Over 42% on NQ with 64 examples, beating a 540B model by 3% with 50× fewer parameters.
- **Asai et al., *Self-RAG* (2023)** — [arXiv:2310.11511](https://arxiv.org/abs/2310.11511). Reflection tokens decide *when* to retrieve and critique what comes back.
- **Schick et al., *Toolformer* (2023)** — [arXiv:2302.04761](https://arxiv.org/abs/2302.04761). Self-supervised learning of API calls (calculator, search, QA).
- **RL for search:**
  - **Search-R1** ([arXiv:2503.09516](https://arxiv.org/abs/2503.09516)): +41% for Qwen2.5-7B and +20% for 3B over RAG baselines.
  - **R1-Searcher** ([arXiv:2503.05592](https://arxiv.org/abs/2503.05592)): outcome-only RL, with no cold start.
  - **ReSearch** ([arXiv:2503.19470](https://arxiv.org/abs/2503.19470)): reflection and self-correction emerge.
- **Kotoge et al., *Can Compact Language Models Search Like Agents?* (DGPO, 2025)** — [arXiv:2508.20324](https://arxiv.org/abs/2508.20324). Plain RL is unstable at 0.5–1B because rewards are sparse. Teacher-guided policy optimization fixes it. **This matters for Phases 3–4:** plan for teacher guidance, not pure RL, at our scale.

## 8. Knowing what you don't know (Track 3)

- **Kadavath et al., *Language Models (Mostly) Know What They Know* (2022)** — [arXiv:2207.05221](https://arxiv.org/abs/2207.05221). Self-evaluation P(True) and a trained P(IK) work. P(IK) rises when relevant source material is in context.
- **Zhang et al., *R-Tuning* (2023)** — [arXiv:2311.09677](https://arxiv.org/abs/2311.09677). Refusal-aware SFT data is built from the gap between what the model knows and the SFT data. Refusal generalizes out of domain. This is the template for our known/unknown labelling.
- **Kalai et al., *Why Language Models Hallucinate* (2025)** — [arXiv:2509.04664](https://arxiv.org/abs/2509.04664). Pretraining creates statistical pressure to err, and benchmarks reward guessing over abstaining. Our Phase 3 metrics must therefore penalize confident errors, not only reward accuracy.
- **When to retrieve:** **Adaptive-RAG** ([arXiv:2403.14403](https://arxiv.org/abs/2403.14403)) routes by query complexity. **SKR** ([arXiv:2310.05002](https://arxiv.org/abs/2310.05002)) retrieves only when self-knowledge says it should, which beats always-retrieve.

## 9. Compact chain-of-thought (Track 4)

- **Hao et al., *Coconut* (2024)** — [arXiv:2412.06769](https://arxiv.org/abs/2412.06769). Reasoning in continuous latent space, with BFS-like search over options.
- **Deng et al., *Stepwise Internalization* (2024)** — [arXiv:2405.14838](https://arxiv.org/abs/2405.14838). GPT-2 Small learns 9×9 multiplication at up to 99% with no visible CoT. Directly relevant at our scale.
- **TokenSkip** ([arXiv:2502.12067](https://arxiv.org/abs/2502.12067)): about 40% fewer reasoning tokens on GSM8K with less than 0.4% accuracy loss. **C3oT** ([arXiv:2412.11664](https://arxiv.org/abs/2412.11664)): more than 50% shorter CoT.
- **Geiping et al., *Recurrent Depth* (2025)** — [arXiv:2502.05171](https://arxiv.org/abs/2502.05171). Test-time compute in latent recurrence instead of tokens.
- **Jolicoeur-Martineau, *Tiny Recursive Model* (2025)** — [arXiv:2510.04871](https://arxiv.org/abs/2510.04871). A 7M-param, 2-layer recursive network gets 45% on ARC-AGI-1. Very small models can reason on narrow tasks.
- **Rizvi-Martel et al., *The Illusion of Superposition?* (2026)** — [arXiv:2604.06374](https://arxiv.org/abs/2604.06374). Only latent-CoT models trained *from scratch* show superposition. Fine-tuned ones fall back on shortcuts. This is a reason to try latent CoT in pretraining, not only as a post-hoc fine-tune.

## 10. Baseline reproduction (Phase 0)

- **Karpathy, *Reproducing GPT-2 (124M) in llm.c* (2024-05-28)** — [llm.c discussion #481](https://github.com/karpathy/llm.c/discussions/481). Trained on 10B FineWeb tokens: HellaSwag 29.9, against 29.4 for the original GPT-2. This is the Phase 0 reference.
- **nanoGPT** ([github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)) is marked deprecated in its README, which points to **nanochat**.
- **modded-nanogpt speedrun** ([github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)): reach 3.28 val loss on FineWeb on 8×H100. It's a source of training-efficiency tricks. The record checked on 2026-10-13 (#92, 2026-08-30) uses a large hashed n-gram table, which is itself a memory-separation trick.

## Open gaps this project targets

1. **Parameter ratio for reasoning.** KLLMs and LMLM show knowledge-light pretraining helps evidence-grounded *factual* tasks, compared at equal model sizes. No source found reports a **score-vs-size curve on multi-step reasoning** (math, multi-hop, synthetic deduction) comparing knowledge-light and normal recipes under matched compute.
2. **Which entities to remove.** KLLMs anonymizes all entities, or a random fraction of them. Removing only *long-tail* entities and keeping head concepts tests where the line between concept and fact falls. Realistic surrogate names vs typed placeholder tags is also untested.
3. **Post-training a knowledge-light base.** Nobody has tested distillation or RL for reasoning, search, or compact CoT on a knowledge-light base.
4. **Context utilization at small scale.** Pandey 2026 shows models of 7B and below largely ignore oracle context. Does a knowledge-light base, which *cannot* fall back on memory, learn to use context better? KLLMs' results hint yes, but only after fine-tuning per benchmark and only on factual tasks. This is untested on reasoning.
5. **Knowledge vs reasoning interaction.** LIMO and SynthWorlds suggest stored knowledge helps reasoning. Neither tests a model *pretrained* without it, with facts supplied at test time.
6. **Calibration from data, not from tuning.** KLLMs reports better abstention as a side effect of anonymization. Phase 3 can test whether this makes "I need to look this up" cheaper to train.
7. **Compact CoT from scratch.** The Rizvi-Martel 2026 finding suggests latent reasoning behaves differently when learned in pretraining. This has not been explored with knowledge-light bases.
