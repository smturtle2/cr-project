#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

target="${TRAIN_TARGET:-tmp_main.py}"
if [[ ! -f "${target}" ]]; then
  echo "train target not found: ${target}" >&2
  echo "set TRAIN_TARGET or create tmp_main.py from tmp_main_base.py" >&2
  exit 1
fi

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

nproc_per_node="${NPROC_PER_NODE:-$(count_visible_cuda_devices)}"
if [[ ! "${nproc_per_node}" =~ ^[0-9]+$ ]]; then
  echo "NPROC_PER_NODE must be a non-negative integer, got: ${nproc_per_node}" >&2
  exit 1
fi

if (( nproc_per_node > 1 )); then
  exec uv run torchrun --standalone --nproc-per-node="${nproc_per_node}" "${target}" "$@"
fi

exec uv run python "${target}" "$@"
