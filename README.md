# Efficient-LM Challenge

**How small can a language model be and still reason at frontier level — if it doesn't have to memorize the world?**

This is an open research project toward a *cognitive core*: a small, efficient language model that keeps reasoning, language understanding and common sense, but deliberately drops encyclopedic knowledge. Facts live outside the model — on the web, in documents, in tools — and the model is trained to know when it doesn't know, and to go find out on its own.

> Frontier models are a student who memorized the whole library *and* learned to think.
> This project builds the student who takes the **open-book exam**: no page numbers memorized, but knows every subject's language, which book to open, and how to solve the problem.

## Core hypothesis

Much of a large model's parameter budget is spent storing facts (roughly 2 bits of knowledge per parameter, per Allen-Zhu & Li). If long-tail facts are removed from pretraining and supplied by retrieval at test time, **the same reasoning ability should be reachable with a much smaller model**.

The main counter-hypothesis to test: reasoning may depend on stored knowledge (LIMO found reasoning is cheap to *elicit* only when domain knowledge was already learned in pretraining). If so, removing facts will hurt reasoning even when retrieval is available.

## The headline experiment

Train two data recipes — **normal** and **knowledge-light** — at several model sizes. Evaluate all models **with retrieval and tools available**. Plot score vs. model size.

- Knowledge-light curve reaches the same score at a smaller size → removing knowledge saves parameters.
- Curves stay parallel → knowledge was doing hidden work for reasoning.

Either result answers the question.

## Research tracks

| # | Track | Question |
|---|---|---|
| 1 | **Data equilibrium** | What is the smallest pretraining mix that keeps full language understanding and common sense? Where is the knee between "stripped too much" and "redundant"? |
| 2 | **Making it smart** | How far do distillation + RL with verifiable rewards take a small knowledge-light base? |
| 3 | **Knowing it doesn't know** | Can the model reliably say "I need to look this up" instead of hallucinating — without over-searching? |
| 4 | **Reasoning / compact CoT** | Can chain-of-thought be made reliable, then compressed into a short internal language without losing accuracy? |
| 5 | **Self-directed lookup** | When it doesn't know, can the model write its own code / queries to find out, verify, and answer — with no human help? |

Tracks 3 and 5 are one system in practice: *notice the gap → write code to find out → verify → answer.*

## Repository layout

```
README.md                    this file
LICENSE                      MIT
CLAUDE.md                    working guide for AI coding agents on this repo
docs/
  RESEARCH_PLAN.md           phases, experiments, metrics, success criteria
  LITERATURE_REVIEW.md       what has been tried, what it showed, open gaps
  EXPERIMENT_PROTOCOL.md     rules that make results trustworthy
experiments/
  _TEMPLATE.md               copy this for every run
```

Code directories (`src/`, `data/`, `evals/`, `configs/`) will be added as Phase 0 begins.

## Status

- [x] Literature review and research plan
- [x] **Phase 0** — reproduce a known GPT-2-small baseline; freeze eval suite; measure noise floor
- [ ] **Phase 1** — data-slice ablations (fixed token budget)
- [ ] **Phase 2** — distillation + RL for reasoning
- [ ] **Phase 3** — "I don't know" calibration + retrieval
- [ ] **Phase 4** — self-directed code/tool lookup
- [ ] **Phase 5** — compact chain-of-thought

## Constraints

Solo-developer project on a single GPU (RTX 5090 class). Everything is designed so that the key decisions can be made at ~150M parameters, where small-scale data rankings are known to predict 1B-scale outcomes about 80% of the time (AI2 DataDecide).

## License

MIT — see [LICENSE](LICENSE).
