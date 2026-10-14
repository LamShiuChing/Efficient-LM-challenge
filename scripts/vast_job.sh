#!/usr/bin/env bash
# Runs ON a vast.ai instance, as its onstart script. Never run it on the dev machine.
#   bash scripts/vast_job.sh bench          # 124M throughput, eager vs compile, several micro-batches
#   bash scripts/vast_job.sh prep           # FineWeb 10B -> shards -> HF dataset repo
#   bash scripts/vast_job.sh rename         # Phase 1a: NER + entity-renamed arms r50 r80 r100 tag100 -> $RENAMED_REPO
#   bash scripts/vast_job.sh train <config> [--key=value ...]
#   bash scripts/vast_job.sh eval <hf:gpt2|ckpt:runs/x/ckpt.pt> <out_name> <dev|test>   (ckpt fetched from HF if absent)
#   bash scripts/vast_job.sh valloss <runs/x/ckpt.pt> <normal|arm> <out_name>              (loss on that val shard)
# Jobs can be chained: `bench prep`. Env: HF_TOKEN (write). MAX_HOURS caps the whole script (default 1).
# DATA_ARM=<arm> makes train fetch that renamed arm into data_cache/<arm>/; point the run at it with
#   '--train_pattern=data_cache/<arm>/fineweb_train_*.bin' '--val_pattern=data_cache/<arm>/fineweb_val_*.bin'
# Whatever happens, the log is uploaded to HF and the instance destroys itself (budget safety).
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0 PIP_BREAK_SYSTEM_PACKAGES=1  # container's system python (PEP 668)
DATA_REPO=lamshiu/efficient-lm-fineweb10B-gpt2   # dataset repo: tokenized shards + manifest.json
RENAMED_REPO=lamshiu/efficient-lm-fineweb-renamed # dataset repo: <arm>/ shards + manifest.json, ner/ spans + entity table
RUNS_REPO=lamshiu/efficient-lm-runs              # model repo: runs/<name>/ logs + checkpoints, jobs/ logs
JOB_ID="$(date -u +%Y%m%dT%H%M%SZ)-${CONTAINER_ID:-local}"
LOG=/workspace/job-$JOB_ID.log                   # outside the repo, so the git tree stays clean
exec > >(tee -a "$LOG") 2>&1

finish() {
  echo "== finish $(date -u)"
  timeout 900 hf upload "$RUNS_REPO" "$LOG" "jobs/$JOB_ID.log" --no-private || echo "log upload failed"
  # REST call, so self-destroy works even if pip installs failed (docs.vast.ai/api-reference/instances/destroy-instance)
  curl -sS -X DELETE "https://console.vast.ai/api/v0/instances/$CONTAINER_ID" -H "Authorization: Bearer $CONTAINER_API_KEY"
}
trap finish EXIT
# ponytail: one wall-clock cap for the whole script; destroying the instance kills everything still running
( sleep "${MAX_HOURS:-1}h"; echo "== MAX_HOURS=${MAX_HOURS:-1} hit"; finish ) &

echo "== $JOB_ID commit $(git rev-parse HEAD) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
pip install -q tiktoken==0.12.0 huggingface_hub==1.22.0 pyarrow numpy || exit 1
# Fetch the gpt2 BPE files once, with retries, before any multiprocessing pool:
# 30 workers each downloading them at the same time timed out.
export TIKTOKEN_CACHE_DIR=/workspace/tiktoken
tok_ok() { python -c "import tiktoken; tiktoken.get_encoding('gpt2')"; }
for i in 1 2 3 4 5; do tok_ok && break; sleep 20; done
tok_ok || exit 1

fetch_data() {  # TRAIN_SHARDS=N fetches val + the first N train shards (100M tokens each); default all
  local repo=$DATA_REPO dir=data_cache/fineweb10B p="" files=()
  if [ -n "${DATA_ARM:-}" ]; then repo=$RENAMED_REPO dir=data_cache p=$DATA_ARM/ files=("$p"); fi  # "<arm>/" = whole folder
  if [ -n "${TRAIN_SHARDS:-}" ]; then
    files=("${p}manifest.json" "${p}fineweb_val_000000.bin")
    for i in $(seq 1 "$TRAIN_SHARDS"); do files+=("$p$(printf 'fineweb_train_%06d.bin' "$i")"); done
  fi
  hf download "$repo" "${files[@]}" --repo-type dataset --local-dir "$dir"
}

fetch_ckpt() {  # fetch_ckpt runs/x/ckpt.pt : download from the runs repo unless already here ("hf:..." is a no-op)
  [[ $1 == runs/* ]] && [ ! -f "$1" ] && { hf download "$RUNS_REPO" "$1" --local-dir . || return 1; }
  return 0
}

while [ $# -gt 0 ]; do
  job=$1; shift
  echo "== job $job $(date -u)"
  case $job in
    bench)
      for c in true false; do for b in 16 32 64; do
        python src/train.py configs/gpt2_124m_fineweb10B.json --synthetic --benchmark_steps=20 \
          --compile=$c --micro_batch_size=$b 2>&1 | grep -E "BENCHMARK|Error|error" | tail -3
      done; done ;;
    prep)
      python data/prepare_fineweb.py || exit 1
      hf upload "$DATA_REPO" data_cache/fineweb10B . --repo-type dataset --no-private || exit 1 ;;
    rename)  # versions verified on PyPI (spacy) and github.com/explosion/spacy-models releases (model wheel sha256).
      # numpy pinned to the image's version (thinc 8.3 accepts numpy>=1.21,<3), so pip never upgrades it
      npv=$(python -c "import numpy; print(numpy.__version__)") || exit 1
      pip install -q spacy==3.8.16 "numpy==$npv" "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85" || exit 1
      python data/rename_entities.py --selftest || exit 1
      python data/rename_entities.py ner || exit 1  # workers sized from the cgroup CPU quota and RAM
      hf upload "$RENAMED_REPO" data_cache/renamed/ner ner --repo-type dataset --no-private || exit 1
      for arm in r50 r80 r100 tag100; do
        python data/rename_entities.py rename --arm $arm || exit 1
        hf upload "$RENAMED_REPO" "data_cache/renamed/$arm" "$arm" --repo-type dataset --no-private || exit 1
      done ;;
    train)
      cfg=$1; shift; args=()
      while [ $# -gt 0 ] && [[ $1 == --* ]]; do args+=("$1"); shift; done
      fetch_data || exit 1
      # torch.compile (triton) builds a small C launcher: needs gcc + Python headers (absent in -runtime image)
      command -v gcc >/dev/null || { apt-get update -qq && apt-get install -y -qq gcc python3-dev >/dev/null; } || exit 1
      name=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['run_name'])" "$cfg")
      for a in "${args[@]}"; do [[ $a == --run_name=* ]] && name=${a#--run_name=}; done
      mkdir -p "runs/$name"
      ( while sleep 1800; do hf upload "$RUNS_REPO" "runs/$name" "runs/$name" --no-private -q; done ) &
      python src/train.py "$cfg" "${args[@]}"; rc=$?
      hf upload "$RUNS_REPO" "runs/$name" "runs/$name" --no-private || exit 1
      [ $rc -eq 0 ] || exit $rc ;;
    eval)  # eval <hf:name|ckpt:path> <out_name> <dev|test>: harness -> runs repo evals/<out_name>.json
      model=$1 out=$2 split=$3; shift 3
      fetch_ckpt "${model#ckpt:}" || exit 1
      pip install -q datasets==5.0.1 transformers==5.13.0 || exit 1
      python -m evals.test_harness || exit 1
      mkdir -p evals_out
      python -m evals.run --model "$model" --split "$split" --out "evals_out/$out.json" || exit 1
      hf upload "$RUNS_REPO" "evals_out/$out.json" "evals/$out.json" --no-private || exit 1 ;;
    valloss)  # valloss <runs/x/ckpt.pt> <normal|arm> <out_name> -> runs repo evals/<out_name>.json
      ck=$1 which=$2 out=$3; shift 3
      fetch_ckpt "$ck" || exit 1
      if [ "$which" = normal ]; then
        hf download "$DATA_REPO" fineweb_val_000000.bin --repo-type dataset --local-dir data_cache/fineweb10B || exit 1
        pat=data_cache/fineweb10B/fineweb_val_000000.bin
      else
        hf download "$RENAMED_REPO" "$which/fineweb_val_000000.bin" --repo-type dataset --local-dir data_cache || exit 1
        pat=data_cache/$which/fineweb_val_000000.bin
      fi
      mkdir -p evals_out
      python -m evals.val_loss "$ck" "$pat" --out "evals_out/$out.json" || exit 1
      hf upload "$RUNS_REPO" "evals_out/$out.json" "evals/$out.json" --no-private || exit 1 ;;
    *) echo "unknown job $job"; exit 1 ;;
  esac
done
