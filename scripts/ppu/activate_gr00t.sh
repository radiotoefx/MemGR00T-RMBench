#!/usr/bin/env bash

# Source this file to use a PPU environment with repository-local writable caches.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source scripts/ppu/activate_gr00t.sh instead of executing it" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GR00T_ENV="${GR00T_ENV:-$REPO_ROOT/.venv}"
CACHE_ROOT="${GR00T_CACHE_DIR:-${XDG_CACHE_HOME:-$REPO_ROOT/.cache}/memgr00t}"

if [[ ! -f "$GR00T_ENV/bin/activate" ]]; then
  echo "GR00T environment not found: $GR00T_ENV" >&2
  return 2
fi
if [[ -z "${GR00T_PPU_DEVICE:-}" ]]; then
  echo "set GR00T_PPU_DEVICE to one or more physical device ids before sourcing" >&2
  return 2
fi
if [[ ! "$GR00T_PPU_DEVICE" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GR00T_PPU_DEVICE must contain comma-separated physical PPU ids" >&2
  return 2
fi

source "$GR00T_ENV/bin/activate"
unset HF_ENDPOINT
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$CACHE_ROOT/python}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"
export GR00T_MODEL_PATH="${GR00T_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
export NO_ALBUMENTATIONS_UPDATE=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" "$TRITON_CACHE_DIR" "$TMPDIR"

export GR00T_PPU_DEVICE
export CUDA_VISIBLE_DEVICES="$GR00T_PPU_DEVICE"
if [[ -z "${CUDA_HOME:-}" && -n "${PPU_SDK:-}" ]]; then
  export CUDA_HOME="$PPU_SDK/CUDA_SDK"
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

unset SCRIPT_DIR REPO_ROOT CACHE_ROOT
