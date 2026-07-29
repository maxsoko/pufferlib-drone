#!/usr/bin/env bash
# Continue a promoted Stage-C recurrent checkpoint through official gate 3.
set -euo pipefail

ROOT="${REPO_ROOT:-$HOME/pufferlib-drone}"
cd "$ROOT"
source .venv/bin/activate
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_ARCH="${NVCC_ARCH:-sm_86}"

LOAD_PATH="${LOAD_PATH:-}"
if [ -z "$LOAD_PATH" ] || [ ! -f "$LOAD_PATH" ]; then
    echo "set LOAD_PATH to a promoted Stage-C checkpoint" >&2
    exit 2
fi
if pgrep -af 'pufferlib.pufferl train' | grep -v "$$" >/dev/null; then
    echo "another PufferLib training process is already running" >&2
    exit 2
fi
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    bash build.sh drone_race
fi

python -m pufferlib.pufferl train drone_race_full_policy_stage_d_gate3 \
    --load-model-path "$LOAD_PATH" \
    --train.total-timesteps "${CHUNK_STEPS:-2000000}"
