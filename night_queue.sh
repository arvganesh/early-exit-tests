#!/usr/bin/env bash
set -euo pipefail

# Runs a sequence of two-stage UltraChat experiments sequentially.
#
# Usage:
#   WAIT_PID=<pid> ./night_queue.sh
# or:
#   ./night_queue.sh
#
# Monitor:
#   tail -f ./logs/night_queue_*.log

cd "$(dirname "$0")"

WAIT_PID="${WAIT_PID:-}"

MODEL_PATH_DEFAULT="/home/persist/hf/local/Llama-3.2-1B-Instruct"
MODEL_PATH="${MODEL_PATH:-$MODEL_PATH_DEFAULT}"

COMMON_ENV=(
  WANDB_MODE=offline
  WANDB_SILENT=true
  WANDB_DIR=/home/persist/wandb
  HF_HOME=/home/persist/hf
  TRANSFORMERS_CACHE=/home/persist/hf
  HF_DATASETS_CACHE=/home/persist/hf
  TORCH_HOME=/home/persist/torch
  MODEL_PATH="$MODEL_PATH"
  OUTPUT_DIR=../checkpoints/models
  EVAL_DIR=../evaluations
  MAX_LENGTH=2048
  KL_TEMP=1.0
  ATTN_IMPL=auto
  TORCH_DTYPE=bf16
  TORCH_COMPILE=none
  EVAL_EVERY=0
  EVAL_MAX_BATCHES=32
  SAVE_EVERY=0
  SPEC_T=0.7
  SPEC_TOP_P=0.95
  SPEC_TOP_K=0
  SPEC_GAMMA=8
  SPEC_NUM_PROMPTS=32
  SPEC_PROMPT_LEN=256
  SPEC_MAX_NEW=64
  QUALITY_MAX_BATCHES=16
)

ts="$(date +%Y%m%d_%H%M%S)"
queue_log="./logs/night_queue_${ts}.log"
mkdir -p ./logs

log() { echo "[$(date +%F' '%T)] $*" | tee -a "$queue_log" ; }

if [[ -n "$WAIT_PID" ]]; then
  log "Waiting for PID $WAIT_PID to exit..."
  while kill -0 "$WAIT_PID" >/dev/null 2>&1; do
    sleep 30
  done
  log "PID $WAIT_PID exited; starting queued experiments."
fi

run_exp() {
  local name="$1"; shift
  local run_ts
  run_ts="$(date +%Y%m%d_%H%M%S)"
  local run_type_base="night_${name}_${run_ts}"
  log "=== START ${name} (RUN_TYPE_BASE=${run_type_base}) ==="

  # Print the command we're about to run.
  {
    echo "+ env ${COMMON_ENV[*]} RUN_TYPE_BASE=${run_type_base} $*"
  } | tee -a "$queue_log"

  env "${COMMON_ENV[@]}" RUN_TYPE_BASE="${run_type_base}" "$@" 2>&1 | tee -a "$queue_log"
  log "=== DONE ${name} (RUN_TYPE_BASE=${run_type_base}) ==="
}

# Experiment 1: deeper exit to test "upper bound" acceptance at near-full depth.
run_exp "ultrachat_L14_const_2k_2k" \
  LAYERS="14" \
  MAX_STEPS_HEAD=2000 BATCH_SIZE_HEAD=2 GRAD_ACCUM_HEAD=8 LR_HEAD=5e-5 LR_SCHEDULE_HEAD=constant \
  MAX_STEPS_FTLAST=2000 BATCH_SIZE_FTLAST=2 GRAD_ACCUM_FTLAST=8 LR_FTLAST=2e-5 LR_SCHEDULE_FTLAST=constant \
  bash run_sweep_ultrachat_two_stage.sh

# Experiment 2: longer optimization at the best-performing mid-late exit.
run_exp "ultrachat_L13_const_4k_4k" \
  LAYERS="13" \
  MAX_STEPS_HEAD=4000 BATCH_SIZE_HEAD=2 GRAD_ACCUM_HEAD=8 LR_HEAD=5e-5 LR_SCHEDULE_HEAD=constant \
  MAX_STEPS_FTLAST=4000 BATCH_SIZE_FTLAST=2 GRAD_ACCUM_FTLAST=8 LR_FTLAST=2e-5 LR_SCHEDULE_FTLAST=constant \
  bash run_sweep_ultrachat_two_stage.sh

log "All queued experiments finished."

