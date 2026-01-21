#!/usr/bin/env bash
set -euo pipefail

# LR/schedule probe at a single exit layer (ℓ=11) with frequent validation,
# including a pre-update validation point at step 0.

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints/models}"

TARGET_LAYER="${TARGET_LAYER:-11}"
MAX_STEPS="${MAX_STEPS:-600}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
KL_TEMP="${KL_TEMP:-1.0}"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR_BASE="${LR_BASE:-2e-5}"
LR_HIGH="${LR_HIGH:-5e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"

EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-32}"

ATTN_IMPL="${ATTN_IMPL:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
TORCH_COMPILE="${TORCH_COMPILE:-none}"

WANDB_PROJECT="${WANDB_PROJECT:-Resurrection-UltraChat}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_TYPE="${RUN_TYPE:-lr_sweep_ultrachat_layer${TARGET_LAYER}_${TS}}"

echo "Run type: ${RUN_TYPE}"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ultrachat"
echo "Layer: ${TARGET_LAYER}"
echo "Max steps: ${MAX_STEPS}"
echo "Eval every: ${EVAL_EVERY}"
echo "W&B project: ${WANDB_PROJECT}"

run_one() {
  local lr_schedule="$1"
  local lr="$2"
  echo "=== lr_schedule=${lr_schedule} lr=${lr} ==="
  python train.py \
    --dataset ultrachat \
    --model_path "${MODEL_PATH}" \
    --target_layer "${TARGET_LAYER}" \
    --loss_type kl_divergence \
    --kl_temperature "${KL_TEMP}" \
    --ft_head \
    --batch_size "${BATCH_SIZE}" \
    --grad_accumulate_steps "${GRAD_ACCUM}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${lr}" \
    --warmup_step_ratio "${WARMUP_RATIO}" \
    --lr_schedule "${lr_schedule}" \
    --max_length "${MAX_LENGTH}" \
    --device cuda \
    --attn_implementation "${ATTN_IMPL}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --torch_compile "${TORCH_COMPILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_type "${RUN_TYPE}" \
    --notes "LR probe (UltraChat). layer=${TARGET_LAYER}, steps=${MAX_STEPS}, lr=${lr}, schedule=${lr_schedule}" \
    --eval_every_steps "${EVAL_EVERY}" \
    --eval_max_batches "${EVAL_MAX_BATCHES}" \
    --wandb \
    --wandb_project "${WANDB_PROJECT}"
}

run_one linear "${LR_BASE}"
run_one constant "${LR_BASE}"
run_one linear "${LR_HIGH}"
run_one constant "${LR_HIGH}"

echo "Done. Check W&B group: ${RUN_TYPE}"

