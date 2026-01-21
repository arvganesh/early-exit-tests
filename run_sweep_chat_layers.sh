#!/usr/bin/env bash
set -euo pipefail

# Coarse exit-layer sweep for a 16-layer LLaMA.
# Runs head-only KL distillation for layers ~{4,8,12,16} (0-based {3,7,11,15}),
# then evaluates checkpoints and writes plots/CSVs.

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints/models}"
EVAL_DIR="${EVAL_DIR:-../evaluations}"

MAX_STEPS="${MAX_STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
KL_TEMP="${KL_TEMP:-1.0}"

ATTN_IMPL="${ATTN_IMPL:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
TORCH_COMPILE="${TORCH_COMPILE:-none}"

EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-8}"

SPEC_PROMPT_LEN="${SPEC_PROMPT_LEN:-256}"
SPEC_MAX_NEW="${SPEC_MAX_NEW:-128}"
SPEC_NUM_PROMPTS="${SPEC_NUM_PROMPTS:-64}"
SPEC_GAMMA="${SPEC_GAMMA:-8}"
SPEC_T="${SPEC_T:-1.0}"
SPEC_TOP_P="${SPEC_TOP_P:-0.95}"
SPEC_TOP_K="${SPEC_TOP_K:-0}"
QUALITY_MAX_BATCHES="${QUALITY_MAX_BATCHES:-32}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_TYPE="${RUN_TYPE:-sweep_chat_${TS}}"
MODEL_FOLDER="${MODEL_PATH#*/}"

# Default: skip full-depth (15) because it's redundant with the teacher baseline and can be
# memory-heavy (this project loads both teacher + draft). Override with LAYERS="3 7 11 15".
LAYERS_STR="${LAYERS:-"3 7 11"}"
read -r -a LAYERS <<< "${LAYERS_STR}"

echo "Run type: ${RUN_TYPE}"
echo "Model: ${MODEL_PATH}"
echo "Layers: ${LAYERS[*]}"

for L in "${LAYERS[@]}"; do
  echo "=== Training target_layer=${L} ==="
  python train.py \
    --dataset sharegpt \
    --no_sharegpt_filter_non_english \
    --model_path "${MODEL_PATH}" \
    --target_layer "${L}" \
    --loss_type kl_divergence \
    --kl_temperature "${KL_TEMP}" \
    --ft_head \
    --batch_size "${BATCH_SIZE}" \
    --grad_accumulate_steps "${GRAD_ACCUM}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LR}" \
    --save_every_steps "${SAVE_EVERY}" \
    --warmup_step_ratio "${WARMUP_RATIO}" \
    --max_length "${MAX_LENGTH}" \
    --device cuda \
    --attn_implementation "${ATTN_IMPL}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --torch_compile "${TORCH_COMPILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_type "${RUN_TYPE}" \
    --notes "Coarse layer sweep for chat/spec decode (head-only KL). layer=${L}, steps=${MAX_STEPS}, max_length=${MAX_LENGTH}" \
    --eval_every_steps "${EVAL_EVERY}" \
    --eval_max_batches "${EVAL_MAX_BATCHES}"
done

CKPT_GLOB="${OUTPUT_DIR}/${MODEL_FOLDER}/${RUN_TYPE}/layer*/model_*.pt"
OUT_DIR="${EVAL_DIR}/${RUN_TYPE}"
echo "=== Evaluating checkpoints ==="
echo "Checkpoint glob: ${CKPT_GLOB}"
echo "Output dir: ${OUT_DIR}"

python evaluate_checkpoints.py \
  --model_path "${MODEL_PATH}" \
  --checkpoint_glob "${CKPT_GLOB}" \
  --out_dir "${OUT_DIR}" \
  --device cuda \
  --attn_implementation "${ATTN_IMPL}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --max_length "${MAX_LENGTH}" \
  --quality_max_batches "${QUALITY_MAX_BATCHES}" \
  --quality_datasets sharegpt \
  --spec_prompt_length "${SPEC_PROMPT_LEN}" \
  --spec_max_new_tokens "${SPEC_MAX_NEW}" \
  --spec_num_prompts "${SPEC_NUM_PROMPTS}" \
  --speculate_len "${SPEC_GAMMA}" \
  --spec_temperature "${SPEC_T}" \
  --spec_top_p "${SPEC_TOP_P}" \
  --spec_top_k "${SPEC_TOP_K}"

echo "Done. Results: ${OUT_DIR}"
