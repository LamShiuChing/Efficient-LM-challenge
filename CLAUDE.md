# CLAUDE.md

Working guide for AI coding agents on this repo.

## What this project is

A research project that asks whether a small LM can reason well if encyclopedic knowledge is removed from pretraining and supplied by retrieval and tools at test time. See [README.md](README.md) for the hypothesis and [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) for the phases.

The deliverable is **trustworthy evidence**, not a model. A clean null result is a success. A result that is impressive but unreproducible is a failure.

## Read first

1. `README.md`: the status checklist says which phase is active.
2. `docs/RESEARCH_PLAN.md`: the section for the active phase.
3. `docs/EXPERIMENT_PROTOCOL.md`: the rules. They are not optional.
4. The newest file in `experiments/`: what was run last and what it concluded.

## Hard rules

- **No run without a committed spec.** Copy `experiments/_TEMPLATE.md` and fill in the hypothesis, prediction, decision rule, config and budget. Commit it, then launch.
- **One change per comparison.** Never change data and model in the same experiment.
- **Never touch test splits** for any decision. Use dev splits.
- **Never edit the frozen sections** of an experiment file after launch. Record changes under *Deviations*.
- **Report honestly.** Give the numbers, including crashes, divergence and nulls. Never re-seed until it works.
- **Caps are explicit.** Each spec states a GPU-hour cap. Stop at the cap and report.
- Commit at milestones: spec frozen, run finished, results written.

## Hardware

- **No local compute.** The user's PC is their daily machine. Do not run training, evals, benchmarks, data prep or tokenizing on it, and not on its GPU or heavily on its CPU. Locally, only edit code and docs, use git, and make light CLI calls.
- **Compute: vast.ai** (`vastai` CLI is configured). Every run happens on a rented instance. Confirm the budget with the user before renting.
- **Data: Hugging Face Hub** (account `lamshiu`). Raw data is pulled from the Hub on the instance. Tokenized shards, eval sets and results that need to persist are pushed to Hub dataset repos, not stored on the PC.

## Long autonomous runs

The user may hand over multi-hour GPU loops and step away. In that mode:

- Freeze a spec before each run, and keep within its cap.
- Log progress to the experiment file so the state can be recovered.
- When a decision is the user's to make (scientific direction, spending money, a public-facing change), stop and leave a clear summary instead of guessing.

## Code conventions

Code does not exist yet. It starts in Phase 0 under `src/`, `data/`, `evals/`, `configs/`. Until conventions are written here:

- Python + PyTorch. Keep it small and readable, nanoGPT-style. One file per concern is fine.
- Every training run is reproducible from a config file plus the git commit.
- Checkpoints, datasets and logs stay out of git.
- Non-trivial logic (data filters, scoring, decontamination) gets one small runnable check.
