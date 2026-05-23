#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

tmux_mode=0
train_args=()
for arg in "$@"; do
  case "${arg}" in
    --tmux)
      tmux_mode=1
      ;;
    *)
      train_args+=("${arg}")
      ;;
  esac
done
set -- "${train_args[@]}"

target="${TRAIN_TARGET:-tmp_main.py}"
uv_bin="${UV:-uv}"

if ! command -v "${uv_bin}" >/dev/null 2>&1; then
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    uv_bin="${HOME}/.local/bin/uv"
  else
    echo "uv is required but was not found on PATH or at ${HOME}/.local/bin/uv" >&2
    exit 1
  fi
fi

show_gpu_list() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
    return
  fi

  echo "nvidia-smi not found; cannot list GPU names." >&2
}

show_usage() {
  cat <<'EOF'
Usage:
  ./train.sh [--tmux] [--gpu ID] [target args...]
  ./train.sh [--tmux] --gpus IDS [target args...]

Options:
  --gpu ID       Run on a single GPU ID, for example: ./train.sh --gpu 0
  --gpus IDS     Run torchrun on selected comma-separated GPU IDs, for example: ./train.sh --gpus 5,6
  --tmux        Start training in a new tmux session.
  -h, --help    Show this help.

Examples:
  ./train.sh --gpu 0
  ./train.sh --tmux --gpu 0
  ./train.sh --gpus 5,6
  ./train.sh --tmux --gpus 5,6
  TRAIN_TARGET=other_runner.py ./train.sh --gpu 0
EOF
}

run_with_sudo_if_needed() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "tmux is not installed and sudo is unavailable" >&2
    exit 1
  fi

  sudo "$@"
}

ensure_tmux() {
  if command -v tmux >/dev/null 2>&1; then
    return
  fi

  if command -v pacman >/dev/null 2>&1; then
    run_with_sudo_if_needed pacman -Syu --needed --noconfirm tmux
  elif command -v apt >/dev/null 2>&1; then
    run_with_sudo_if_needed apt update
    run_with_sudo_if_needed apt install -y tmux
  else
    echo "tmux is not installed and no supported package manager was found" >&2
    echo "install tmux manually, then rerun: ./train.sh --tmux" >&2
    exit 1
  fi

  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux installation finished, but tmux is still not on PATH" >&2
    exit 1
  fi
}

next_tmux_session_name() {
  local existing=""
  local count=0
  local name

  if existing="$(tmux list-sessions -F '#S' 2>/dev/null)"; then
    if [[ -n "${existing}" ]]; then
      count="$(printf '%s\n' "${existing}" | wc -l | tr -d ' ')"
    fi
  fi

  while :; do
    count=$((count + 1))
    name="train-${count}"
    if ! tmux has-session -t "${name}" 2>/dev/null; then
      echo "${name}"
      return
    fi
  done
}

start_tmux_training() {
  ensure_tmux

  local session_name
  local child_command
  local quoted
  local env_name
  session_name="$(next_tmux_session_name)"
  printf -v child_command 'TRAIN_TMUX_CHILD=1 TRAIN_TARGET=%q' "${target}"

  for env_name in UV CUDA_VISIBLE_DEVICES HF_TOKEN NCCL_P2P_DISABLE; do
    if [[ -v "${env_name}" ]]; then
      printf -v quoted ' %s=%q' "${env_name}" "${!env_name}"
      child_command+="${quoted}"
    fi
  done

  child_command+=" ./train.sh"

  for arg in "$@"; do
    printf -v quoted ' %q' "${arg}"
    child_command+="${quoted}"
  done

  echo "starting tmux session: ${session_name}"
  echo "attach later: tmux attach -t ${session_name}"
  echo "command: ${child_command}"
  exec tmux new-session -s "${session_name}" -c "${PWD}" "${child_command}"
}

if (( tmux_mode )) && [[ "${TRAIN_TMUX_CHILD:-0}" != "1" ]]; then
  start_tmux_training "$@"
  exit 0
fi

count_visible_cuda_devices() {
  if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES}" || "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
      echo 0
      return
    fi
    "${uv_bin}" run python - <<'PY'
import os

devices = [item.strip() for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
print(sum(1 for item in devices if item))
PY
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l | tr -d ' '
    return
  fi

  "${uv_bin}" run python - <<'PY'
import torch

print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

count_system_cuda_devices() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l | tr -d ' '
    return
  fi

  count_visible_cuda_devices
}

count_csv_items() {
  local value="$1"
  local commas="${value//[^,]/}"
  echo $((${#commas} + 1))
}

gpu_provided=0
gpu_list_provided=0
gpu_id=""
gpu_list=""
target_args=()

set_gpu_id() {
  if (( gpu_provided )); then
    echo "--gpu can only be provided once." >&2
    exit 1
  fi
  gpu_provided=1
  gpu_id="$1"
}

set_gpu_list() {
  if (( gpu_list_provided )); then
    echo "--gpus can only be provided once." >&2
    exit 1
  fi
  gpu_list_provided=1
  gpu_list="$1"
}

while (($#)); do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    --gpu)
      if (($# < 2)); then
        echo "--gpu requires a GPU ID." >&2
        show_usage >&2
        exit 1
      fi
      set_gpu_id "$2"
      shift 2
      ;;
    --gpu=*)
      set_gpu_id "${1#--gpu=}"
      shift
      ;;
    --gpus)
      if (($# < 2)); then
        echo "--gpus requires comma-separated GPU IDs." >&2
        show_usage >&2
        exit 1
      fi
      set_gpu_list "$2"
      shift 2
      ;;
    --gpus=*)
      set_gpu_list "${1#--gpus=}"
      shift
      ;;
    --)
      shift
      target_args+=("$@")
      break
      ;;
    *)
      target_args+=("$1")
      shift
      ;;
  esac
done

if (( gpu_provided && gpu_list_provided )); then
  echo "--gpu and --gpus cannot be combined." >&2
  show_usage >&2
  exit 1
fi

if (( gpu_provided )) && [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
  echo "--gpu must be a non-negative integer GPU ID, got: ${gpu_id}" >&2
  show_usage >&2
  exit 1
fi

if (( gpu_list_provided )) && [[ ! "${gpu_list}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--gpus must be comma-separated non-negative integer GPU IDs, got: ${gpu_list}" >&2
  show_usage >&2
  exit 1
fi

if (( gpu_provided )); then
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
fi

if (( gpu_list_provided )); then
  export CUDA_VISIBLE_DEVICES="${gpu_list}"
fi

if (( gpu_list_provided )); then
  nproc_per_node="$(count_csv_items "${gpu_list}")"
  if [[ ! "${nproc_per_node}" =~ ^[0-9]+$ ]]; then
    echo "GPU count must be a non-negative integer, got: ${nproc_per_node}" >&2
    exit 1
  fi
  if (( nproc_per_node < 1 )); then
    echo "--gpus requires at least one GPU ID." >&2
    exit 1
  fi
  if [[ ! -f "${target}" ]]; then
    echo "train target not found: ${target}" >&2
    echo "set TRAIN_TARGET or create tmp_main.py from tmp_main_base.py" >&2
    exit 1
  fi
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  exec "${uv_bin}" run torchrun --standalone --nproc-per-node="${nproc_per_node}" "${target}" "${target_args[@]}"
fi

if (( gpu_provided )); then
  if [[ ! -f "${target}" ]]; then
    echo "train target not found: ${target}" >&2
    echo "set TRAIN_TARGET or create tmp_main.py from tmp_main_base.py" >&2
    exit 1
  fi
  exec "${uv_bin}" run python "${target}" "${target_args[@]}"
fi

system_cuda_devices="$(count_system_cuda_devices)"
if [[ ! "${system_cuda_devices}" =~ ^[0-9]+$ ]]; then
  echo "system CUDA device count must be a non-negative integer, got: ${system_cuda_devices}" >&2
  exit 1
fi

if (( ! gpu_provided && system_cuda_devices > 1 )); then
  echo "Multiple GPUs are available. Choose one GPU with --gpu or selected GPUs with --gpus." >&2
  echo >&2
  echo "Detected GPUs:" >&2
  show_gpu_list >&2
  echo >&2
  show_usage >&2
  exit 1
fi

if [[ ! -f "${target}" ]]; then
  echo "train target not found: ${target}" >&2
  echo "set TRAIN_TARGET or create tmp_main.py from tmp_main_base.py" >&2
  exit 1
fi

exec "${uv_bin}" run python "${target}" "${target_args[@]}"
