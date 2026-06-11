#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python_bin="${PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    python_bin=".venv/bin/python"
  else
    python_bin="python3"
  fi
fi

host="${HOST:-127.0.0.1}"
port="${PORT:-8797}"
model_path="${MODEL_PATH:-${CHECKPOINT:-artifacts/3.CLEAR-Net/v4/best.pt}}"

if (($#)) && [[ "$1" != -* ]]; then
  model_path="$1"
  shift
fi

browser_args=()
if [[ "${OPEN_BROWSER:-0}" != "1" ]]; then
  browser_args+=(--no-open-browser)
fi

exec "${python_bin}" artifacts/0.show/clear_net_scene_demo.py \
  --host "${host}" \
  --port "${port}" \
  --model-path "${model_path}" \
  "${browser_args[@]}" \
  "$@"
