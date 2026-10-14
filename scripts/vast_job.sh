#!/usr/bin/env bash
# Runs ON a vast.ai instance, as its onstart script. Never run it on the dev machine.
#   bash scripts/vast_job.sh bench          # 124M throughput, eager vs compile, several micro-batches
#   bash scripts/vast_job.sh prep           # FineWeb 10B -> shards -> HF dataset repo
#   bash scripts/vast_job.sh train <config> [--key=value ...]
# Jobs can be chained: `bench prep`. Env: HF_TOKEN (write). MAX_HOURS caps the whole script (default 1).
# Whatever happens, the log is uploaded to HF and the instance destroys itself (budget safety).
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0 PIP_BREAK_SYSTEM_PACKAGES=1  # container's system python (PEP 668)
DATA_REPO=lamshiu/efficient-lm-fineweb10B-gpt2   # dataset repo: tokenized shards + manifest.json
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

fetch_data() {
  hf download "$DATA_REPO" --repo-type dataset --local-dir data_cache/fineweb10B
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
    train)
      cfg=$1; shift; args=()
      while [ $# -gt 0 ] && [[ $1 == --* ]]; do args+=("$1"); shift; done
      fetch_data || exit 1
      name=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['run_name'])" "$cfg")
      for a in "${args[@]}"; do [[ $a == --run_name=* ]] && name=${a#--run_name=}; done
      mkdir -p "runs/$name"
      ( while sleep 1800; do hf upload "$RUNS_REPO" "runs/$name" "runs/$name" --no-private -q; done ) &
      python src/train.py "$cfg" "${args[@]}"; rc=$?
      hf upload "$RUNS_REPO" "runs/$name" "runs/$name" --no-private || exit 1
      [ $rc -eq 0 ] || exit $rc ;;
    *) echo "unknown job $job"; exit 1 ;;
  esac
done
