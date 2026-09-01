#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEVICE="${GR00T_SERVER_DEVICE:?set the authorized physical PPU id}"
CHECKPOINT="${GR00T_SERVER_CHECKPOINT:?set the GR00T model or compact checkpoint}"
PROCESSOR_PATH="${GR00T_SERVER_PROCESSOR_PATH:-}"
PORT="${GR00T_SERVER_PORT:-5555}"
HOST="${GR00T_SERVER_HOST:-0.0.0.0}"
POLICY_SEED="${GR00T_SERVER_POLICY_SEED:?set the private policy RNG seed}"
NOISE_MODE="${GR00T_SERVER_NOISE_MODE:-independent}"
MANIFEST_PATH="${GR00T_SERVER_MANIFEST_PATH:?set a new launch manifest JSON path}"
STATE_GRIPPER_SEMANTICS="${GR00T_SERVER_STATE_GRIPPER_SEMANTICS:-physical}"
PARENT_CHECKPOINT_SHA256="${GR00T_SERVER_PARENT_CHECKPOINT_SHA256:-}"
MODALITY_CONFIG_PATH="${GR00T_SERVER_MODALITY_CONFIG_PATH:-}"

if [[ ! "$DEVICE" =~ ^[0-9]+$ ]]; then
  echo "GR00T_SERVER_DEVICE must identify exactly one physical PPU; got: $DEVICE" >&2
  exit 2
fi

export GR00T_PPU_DEVICE="$DEVICE"
source "$SCRIPT_DIR/activate_gr00t.sh"
BASE_MODEL_PATH="${GR00T_SERVER_BASE_MODEL_PATH:-${GR00T_MODEL_PATH:-nvidia/GR00T-N1.7-3B}}"
cd "$REPO_ROOT"

ARGS=(
  --model-path "$CHECKPOINT"
  --base-model-path "$BASE_MODEL_PATH"
  --embodiment-tag NEW_EMBODIMENT
  --device cuda:0
  --host "$HOST"
  --port "$PORT"
  --ppu-id "$DEVICE"
  --policy-seed "$POLICY_SEED"
  --noise-mode "$NOISE_MODE"
  --launch-manifest-path "$MANIFEST_PATH"
  --state-arm-semantics actual_qpos
  --state-gripper-semantics "$STATE_GRIPPER_SEMANTICS"
  --arm-action-semantics absolute
  --gripper-action-semantics absolute
  --decoded-absolute-action-boundary
  --strict
)
if [[ -n "$PROCESSOR_PATH" ]]; then
  ARGS+=(--processor-path "$PROCESSOR_PATH")
fi
if [[ -n "$PARENT_CHECKPOINT_SHA256" ]]; then
  ARGS+=(--parent-checkpoint-sha256 "$PARENT_CHECKPOINT_SHA256")
fi
if [[ -n "$MODALITY_CONFIG_PATH" ]]; then
  ARGS+=(--modality-config-path "$MODALITY_CONFIG_PATH")
fi

# Compact checkpoints already carry the trained TTT architecture in config.json;
# do not pass --use-ttt/model overrides, which would invalidate exact loading.
CUDA_VISIBLE_DEVICES="$DEVICE" python gr00t/eval/run_gr00t_server.py "${ARGS[@]}"
