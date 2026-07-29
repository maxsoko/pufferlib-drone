#!/usr/bin/env bash
# Bounded six-gate bootstrap from the proven official three-gate recurrent
# checkpoint. The source policy is expanded with zero encoder columns so its
# initial behavior is preserved under the new progress adapter (measured action
# drift from FP32 layout/dot-product width is below 7e-7).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_ARCH="${NVCC_ARCH:-sm_86}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/mnt/c/Users/anon/code/pufferlib-drone/checkpoints/drone_race_full_policy_gate3_visual/v6c_latest_obsmatch_r135_dropout_e10.bin}"
EXPANDED_CHECKPOINT="${EXPANDED_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/expanded_gate3_parent_input32_fp32.bin}"
TIMESTEPS="${TIMESTEPS:-1000000}"

if [ ! -f "$SOURCE_CHECKPOINT" ]; then
    echo "missing source checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 2
fi
if pgrep -af 'pufferlib.pufferl train' | grep -v "$$" >/dev/null; then
    echo "another PufferLib training process is already running" >&2
    exit 2
fi

python scripts/resize_policy_checkpoint_input.py \
    "$SOURCE_CHECKPOINT" "$EXPANDED_CHECKPOINT" \
    --source-input-dim 23 --target-input-dim 32 \
    --source-precision-bytes 2 --target-precision-bytes 4

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    bash build.sh drone_race --float
fi

LOAD_PATH="${LOAD_PATH:-$EXPANDED_CHECKPOINT}"
if [ ! -f "$LOAD_PATH" ]; then
    echo "missing training load checkpoint: $LOAD_PATH" >&2
    exit 2
fi

python -m pufferlib.pufferl train drone_race_full_policy_six_gate_bootstrap \
    --load-model-path "$LOAD_PATH" \
    --train.total-timesteps "$TIMESTEPS" \
    "$@"
