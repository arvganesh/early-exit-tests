#!/usr/bin/env bash
set -euo pipefail

# Layer sweep (UltraChat) with the best LR setup discovered in the layer-11 LR probe:
# - KL distillation, head-only fine-tuning
# - constant LR = 5e-5
# - max_steps = 600
#
# Launches each layer run in the background (tmux if available; otherwise nohup).
# Note: KL distillation over the full vocab at max_length=4096 is memory-heavy (B*T*V).
# Running multiple layers concurrently on a single 80GB GPU can OOM; run one layer at a time
# (set `LAYERS="3"` etc) or reduce `BATCH_SIZE` and increase `GRAD_ACCUM`.

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-../checkpoints/models}"

# Exit layers are 0-based indices into `.model.layers` (Llama-3.2-1B has 16 layers: 0..15).
LAYERS="${LAYERS:-3}"

MAX_STEPS="${MAX_STEPS:-600}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
KL_TEMP="${KL_TEMP:-1.0}"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"

EVAL_EVERY="${EVAL_EVERY:-100}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-32}"

ATTN_IMPL="${ATTN_IMPL:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
TORCH_COMPILE="${TORCH_COMPILE:-none}"

WANDB_PROJECT="${WANDB_PROJECT:-Resurrection-UltraChat-LayerSweep}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_TYPE="${RUN_TYPE:-layer_sweep_ultrachat_fthead_lr${LR}_steps${MAX_STEPS}_${TS}}"

echo "Run type: ${RUN_TYPE}"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ultrachat"
echo "Layers: ${LAYERS}"
echo "Max steps: ${MAX_STEPS}"
echo "LR schedule: constant (warmup ${WARMUP_RATIO})"
echo "LR: ${LR}"
echo "W&B project: ${WANDB_PROJECT}"

for layer in ${LAYERS}; do
  session="ultrachat_l${layer}_lr${LR}_s${MAX_STEPS}"
  echo "=== Launching layer=${layer} (session=${session}) ==="
  ./launch_train_bg.sh "${session}" python train.py \
    --dataset ultrachat \
    --model_path "${MODEL_PATH}" \
    --target_layer "${layer}" \
    --loss_type kl_divergence \
    --kl_temperature "${KL_TEMP}" \
    --ft_head \
    --batch_size "${BATCH_SIZE}" \
    --grad_accumulate_steps "${GRAD_ACCUM}" \
    --max_steps "${MAX_STEPS}" \
    --learning_rate "${LR}" \
    --warmup_step_ratio "${WARMUP_RATIO}" \
    --lr_schedule constant \
    --max_length "${MAX_LENGTH}" \
    --device cuda \
    --attn_implementation "${ATTN_IMPL}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --torch_compile "${TORCH_COMPILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_type "${RUN_TYPE}" \
    --notes "Layer sweep (UltraChat). layer=${layer}, steps=${MAX_STEPS}, lr=${LR}, schedule=constant" \
    --eval_every_steps "${EVAL_EVERY}" \
    --eval_max_batches "${EVAL_MAX_BATCHES}" \
    --wandb \
    --wandb_project "${WANDB_PROJECT}"
done

echo "Done launching. Logs: ./logs/"
