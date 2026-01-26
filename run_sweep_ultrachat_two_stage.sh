#!/usr/bin/env bash
set -euo pipefail

# Two-stage UltraChat sweep:
#   (1) Head-only KL distillation at multiple exit layers
#   (2) Continue from the tuned head, unfreeze the last retained transformer block, and finetune
#
# Writes evaluation CSVs + paper-ready plots for each stage.

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints/models}"
EVAL_DIR="${EVAL_DIR:-../evaluations}"

# Stage 1 (head-only) defaults: match the original dryrun settings.
MAX_STEPS_HEAD="${MAX_STEPS_HEAD:-200}"
BATCH_SIZE_HEAD="${BATCH_SIZE_HEAD:-4}"
GRAD_ACCUM_HEAD="${GRAD_ACCUM_HEAD:-4}"
LR_HEAD="${LR_HEAD:-2e-5}"
LR_SCHEDULE_HEAD="${LR_SCHEDULE_HEAD:-linear}"

# Stage 2 (head + last block)
MAX_STEPS_FTLAST="${MAX_STEPS_FTLAST:-200}"
BATCH_SIZE_FTLAST="${BATCH_SIZE_FTLAST:-4}"
GRAD_ACCUM_FTLAST="${GRAD_ACCUM_FTLAST:-4}"
LR_FTLAST="${LR_FTLAST:-2e-5}"
LR_SCHEDULE_FTLAST="${LR_SCHEDULE_FTLAST:-linear}"

WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
KL_TEMP="${KL_TEMP:-1.0}"
ULTRACHAT_DATE_STRING="${ULTRACHAT_DATE_STRING:-2026-01-20}"

ATTN_IMPL="${ATTN_IMPL:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
TORCH_COMPILE="${TORCH_COMPILE:-none}"

EVAL_EVERY="${EVAL_EVERY:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-32}"
SAVE_EVERY="${SAVE_EVERY:-0}"

SPEC_PROMPT_LEN="${SPEC_PROMPT_LEN:-256}"
SPEC_MAX_NEW="${SPEC_MAX_NEW:-128}"
SPEC_NUM_PROMPTS="${SPEC_NUM_PROMPTS:-64}"
SPEC_GAMMA="${SPEC_GAMMA:-8}"
SPEC_T="${SPEC_T:-1.0}"
SPEC_TOP_P="${SPEC_TOP_P:-0.95}"
SPEC_TOP_K="${SPEC_TOP_K:-0}"
QUALITY_MAX_BATCHES="${QUALITY_MAX_BATCHES:-32}"

WANDB_PROJECT="${WANDB_PROJECT:-Resurrection-UltraChat}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_TYPE_BASE="${RUN_TYPE_BASE:-sweep_ultrachat_${TS}}"
RUN_TYPE_HEAD="${RUN_TYPE_HEAD:-${RUN_TYPE_BASE}_head}"
RUN_TYPE_FTLAST="${RUN_TYPE_FTLAST:-${RUN_TYPE_BASE}_ftlast}"

if [[ "${MODEL_PATH}" == */* ]]; then
  MODEL_FOLDER="$(basename "${MODEL_PATH}")"
else
  MODEL_FOLDER="${MODEL_PATH#*/}"
fi

LAYERS_STR="${LAYERS:-"3 7 11"}"
read -r -a LAYERS <<< "${LAYERS_STR}"

echo "Model: ${MODEL_PATH}"
echo "Layers: ${LAYERS[*]}"
echo "Stage1 run_type: ${RUN_TYPE_HEAD}"
echo "Stage2 run_type: ${RUN_TYPE_FTLAST}"
echo "W&B project: ${WANDB_PROJECT}"
echo "Stage1 lr_schedule: ${LR_SCHEDULE_HEAD} (warmup ${WARMUP_RATIO})"
echo "Stage2 lr_schedule: ${LR_SCHEDULE_FTLAST} (warmup ${WARMUP_RATIO})"

for L in "${LAYERS[@]}"; do
  echo "=== [Stage1 head-only] target_layer=${L} ==="
  python train.py \
    --dataset ultrachat \
    --model_path "${MODEL_PATH}" \
    --target_layer "${L}" \
    --loss_type kl_divergence \
    --kl_temperature "${KL_TEMP}" \
    --ft_head \
    --batch_size "${BATCH_SIZE_HEAD}" \
    --grad_accumulate_steps "${GRAD_ACCUM_HEAD}" \
    --max_steps "${MAX_STEPS_HEAD}" \
    --learning_rate "${LR_HEAD}" \
    --lr_schedule "${LR_SCHEDULE_HEAD}" \
    --save_every_steps "${SAVE_EVERY}" \
    --eval_every_steps "${EVAL_EVERY}" \
    --eval_max_batches "${EVAL_MAX_BATCHES}" \
    --warmup_step_ratio "${WARMUP_RATIO}" \
    --max_length "${MAX_LENGTH}" \
    --device cuda \
    --attn_implementation "${ATTN_IMPL}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --torch_compile "${TORCH_COMPILE}" \
    --ultrachat_date_string "${ULTRACHAT_DATE_STRING}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_type "${RUN_TYPE_HEAD}" \
    --notes "UltraChat head-only KL. layer=${L}, steps=${MAX_STEPS_HEAD}, max_length=${MAX_LENGTH}, klT=${KL_TEMP}" \
    --wandb \
    --wandb_project "${WANDB_PROJECT}"

  # Pick the final checkpoint for this layer (step = MAX_STEPS_HEAD-1).
  step_last="$((MAX_STEPS_HEAD - 1))"
  shopt -s nullglob
  ckpts=( "${OUTPUT_DIR}/${MODEL_FOLDER}/${RUN_TYPE_HEAD}/layer${L}_${MAX_STEPS_HEAD}steps_begin"*"/model_${step_last}_"*.pt )
  shopt -u nullglob
  if [[ ${#ckpts[@]} -eq 0 ]]; then
    echo "ERROR: Could not find stage1 checkpoint for layer=${L}" >&2
    exit 1
  fi
  CKPT_PATH="${ckpts[-1]}"
  echo "Stage1 checkpoint: ${CKPT_PATH}"

  echo "=== [Stage2 head + last block] target_layer=${L} ==="
  python train.py \
    --dataset ultrachat \
    --model_path "${MODEL_PATH}" \
    --target_layer "${L}" \
    --loss_type kl_divergence \
    --kl_temperature "${KL_TEMP}" \
    --ft_head \
    --ft_last_transformer \
    --checkpoint_path "${CKPT_PATH}" \
    --batch_size "${BATCH_SIZE_FTLAST}" \
    --grad_accumulate_steps "${GRAD_ACCUM_FTLAST}" \
    --max_steps "${MAX_STEPS_FTLAST}" \
    --learning_rate "${LR_FTLAST}" \
    --lr_schedule "${LR_SCHEDULE_FTLAST}" \
    --save_every_steps "${SAVE_EVERY}" \
    --eval_every_steps "${EVAL_EVERY}" \
    --eval_max_batches "${EVAL_MAX_BATCHES}" \
    --warmup_step_ratio "${WARMUP_RATIO}" \
    --max_length "${MAX_LENGTH}" \
    --device cuda \
    --attn_implementation "${ATTN_IMPL}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --torch_compile "${TORCH_COMPILE}" \
    --ultrachat_date_string "${ULTRACHAT_DATE_STRING}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_type "${RUN_TYPE_FTLAST}" \
    --notes "UltraChat head+lastblock KL (init from tuned head). layer=${L}, steps=${MAX_STEPS_FTLAST}, max_length=${MAX_LENGTH}, klT=${KL_TEMP}" \
    --wandb \
    --wandb_project "${WANDB_PROJECT}"
done

echo "=== Evaluating Stage1 checkpoints ==="
CKPT_GLOB_HEAD="${OUTPUT_DIR}/${MODEL_FOLDER}/${RUN_TYPE_HEAD}/layer*/model_*.pt"
OUT_DIR_HEAD="${EVAL_DIR}/${RUN_TYPE_HEAD}"
python evaluate_checkpoints.py \
  --model_path "${MODEL_PATH}" \
  --checkpoint_glob "${CKPT_GLOB_HEAD}" \
  --out_dir "${OUT_DIR_HEAD}" \
  --device cuda \
  --attn_implementation "${ATTN_IMPL}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --max_length "${MAX_LENGTH}" \
  --quality_max_batches "${QUALITY_MAX_BATCHES}" \
  --quality_datasets ultrachat \
  --spec_prompt_dataset ultrachat \
  --spec_prompt_length "${SPEC_PROMPT_LEN}" \
  --spec_max_new_tokens "${SPEC_MAX_NEW}" \
  --spec_num_prompts "${SPEC_NUM_PROMPTS}" \
  --speculate_len "${SPEC_GAMMA}" \
  --spec_temperature "${SPEC_T}" \
  --spec_top_p "${SPEC_TOP_P}" \
  --spec_top_k "${SPEC_TOP_K}"

echo "=== Evaluating Stage2 checkpoints ==="
CKPT_GLOB_FTLAST="${OUTPUT_DIR}/${MODEL_FOLDER}/${RUN_TYPE_FTLAST}/layer*/model_*.pt"
OUT_DIR_FTLAST="${EVAL_DIR}/${RUN_TYPE_FTLAST}"
python evaluate_checkpoints.py \
  --model_path "${MODEL_PATH}" \
  --checkpoint_glob "${CKPT_GLOB_FTLAST}" \
  --out_dir "${OUT_DIR_FTLAST}" \
  --device cuda \
  --attn_implementation "${ATTN_IMPL}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --max_length "${MAX_LENGTH}" \
  --quality_max_batches "${QUALITY_MAX_BATCHES}" \
  --quality_datasets ultrachat \
  --spec_prompt_dataset ultrachat \
  --spec_prompt_length "${SPEC_PROMPT_LEN}" \
  --spec_max_new_tokens "${SPEC_MAX_NEW}" \
  --spec_num_prompts "${SPEC_NUM_PROMPTS}" \
  --speculate_len "${SPEC_GAMMA}" \
  --spec_temperature "${SPEC_T}" \
  --spec_top_p "${SPEC_TOP_P}" \
  --spec_top_k "${SPEC_TOP_K}"

echo "Done."
echo "Stage1 eval: ${OUT_DIR_HEAD}"
echo "Stage2 eval: ${OUT_DIR_FTLAST}"
