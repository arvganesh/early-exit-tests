#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./launch_train_bg.sh <session_name> <command...>

Examples:
  ./launch_train_bg.sh dryrun_h100 python train.py --dataset toy --model_path hf-internal-testing/tiny-random-LlamaForCausalLM --target_layer 1 --loss_type perplexity --ft_head --batch_size 1 --max_steps 5 --learning_rate 2e-5 --max_length 128 --device cuda --output_dir ../checkpoints/models --run_type smoke --notes "smoke"

Notes:
  - If `tmux` is available, this runs in a detached tmux session and logs to `./logs/`.
  - Otherwise it falls back to `nohup` and logs to `./logs/`.
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

session="$1"
shift

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${session}" || "${session}" == "-" ]]; then
  session="train_${timestamp}"
fi

log_dir="${LOG_DIR:-./logs}"
mkdir -p "${log_dir}"
log_file="${log_dir}/${session}_${timestamp}.log"

cmd=( "$@" )
cmd_quoted="$(printf '%q ' "${cmd[@]}")"
pwd_quoted="$(printf '%q' "$(pwd)")"
log_quoted="$(printf '%q' "${log_file}")"
runner_prefix=""
if command -v stdbuf >/dev/null 2>&1; then
  runner_prefix="stdbuf -oL -eL "
fi

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "${session}" "cd ${pwd_quoted} && ${runner_prefix}env PYTHONUNBUFFERED=1 ${cmd_quoted} 2>&1 | tee -a ${log_quoted}"
  echo "Started tmux session: ${session}"
  echo "Attach: tmux attach -t ${session}"
  echo "Log: ${log_file}"
  exit 0
fi

nohup bash -lc "cd ${pwd_quoted} && ${runner_prefix}env PYTHONUNBUFFERED=1 ${cmd_quoted}" >"${log_file}" 2>&1 &
pid=$!
echo "Started background process PID: ${pid}"
echo "Log: ${log_file}"
echo "Follow: tail -f ${log_file}"
