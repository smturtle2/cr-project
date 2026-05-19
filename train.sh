#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

target="${TRAIN_TARGET:-tmp_main.py}"

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
  ./train.sh [--gpu ID] [target args...]
  ./train.sh --multi-gpu [target args...]

Options:
  --gpu ID       Run on a single GPU ID, for example: ./train.sh --gpu 0
  --multi-gpu   Enable multi-GPU training with torchrun.
  -h, --help    Show this help.

Examples:
  ./train.sh --gpu 0
  ./train.sh --multi-gpu
  TRAIN_TARGET=other_runner.py ./train.sh --gpu 0
EOF
}

count_visible_cuda_devices() {
  if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES}" || "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
      echo 0
      return
    fi
    python - <<'PY'
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

  uv run python - <<'PY'
import torch

print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

multi_gpu=0
gpu_provided=0
gpu_id=""
target_args=()

set_gpu_id() {
  if (( gpu_provided )); then
    echo "--gpu can only be provided once." >&2
    exit 1
  fi
  gpu_provided=1
  gpu_id="$1"
}

while (($#)); do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    --multi-gpu)
      multi_gpu=1
      shift
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

if (( gpu_provided && multi_gpu )); then
  echo "--gpu selects one GPU and cannot be combined with --multi-gpu." >&2
  show_usage >&2
  exit 1
fi

if (( gpu_provided )) && [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
  echo "--gpu must be a non-negative integer GPU ID, got: ${gpu_id}" >&2
  show_usage >&2
  exit 1
fi

if (( gpu_provided )); then
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
fi

visible_cuda_devices="$(count_visible_cuda_devices)"
if [[ ! "${visible_cuda_devices}" =~ ^[0-9]+$ ]]; then
  echo "visible CUDA device count must be a non-negative integer, got: ${visible_cuda_devices}" >&2
  exit 1
fi

nproc_per_node="${NPROC_PER_NODE:-${visible_cuda_devices}}"
if [[ ! "${nproc_per_node}" =~ ^[0-9]+$ ]]; then
  echo "NPROC_PER_NODE must be a non-negative integer, got: ${nproc_per_node}" >&2
  exit 1
fi

if (( multi_gpu )); then
  if (( nproc_per_node < 1 )); then
    echo "--multi-gpu requires at least one visible CUDA device." >&2
    exit 1
  fi
  if [[ ! -f "${target}" ]]; then
    echo "train target not found: ${target}" >&2
    echo "set TRAIN_TARGET or create tmp_main.py from tmp_main_base.py" >&2
    exit 1
  fi
  exec uv run torchrun --standalone --nproc-per-node="${nproc_per_node}" "${target}" "${target_args[@]}"
fi

if (( visible_cuda_devices > 1 )); then
  echo "Multiple GPUs are visible. Choose one GPU explicitly or pass --multi-gpu." >&2
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

exec uv run python "${target}" "${target_args[@]}"
