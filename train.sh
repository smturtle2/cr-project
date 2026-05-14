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
if [[ ! -f "${target}" ]]; then
  echo "train target not found: ${target}" >&2
  echo "set TRAIN_TARGET or create tmp_main.py from tmp_main_base.py" >&2
  exit 1
fi

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

  for env_name in NPROC_PER_NODE CUDA_VISIBLE_DEVICES HF_TOKEN; do
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
