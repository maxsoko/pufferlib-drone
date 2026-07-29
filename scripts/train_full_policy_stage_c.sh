#!/usr/bin/env bash
# Stage C: continue the recurrent mean policy through gates 1-2 with no reset.
set -euo pipefail

ROOT="${REPO_ROOT:-$HOME/pufferlib-drone}"
cd "$ROOT"
source .venv/bin/activate
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_ARCH="${NVCC_ARCH:-sm_86}"

if pgrep -af 'pufferlib.pufferl train' | grep -v "$$" >/dev/null; then
    echo "another PufferLib training process is already running" >&2
    exit 2
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    bash build.sh drone_race
fi

# 1024 x 1800 = 1.84M transitions per maximum-length 30-second episode.
# A 2M chunk therefore contains terminal credit while retaining broad parallelism.
REPO_ROOT="$ROOT" \
ENV_NAME=drone_race_full_policy_stage_c \
CHUNK_STEPS="${CHUNK_STEPS:-2000000}" \
MAX_CHUNKS="${MAX_CHUNKS:-6}" \
EVAL_EPISODES="${EVAL_EPISODES:-4096}" \
SUCCESS_TARGET="${SUCCESS_TARGET:-0.90}" \
CRASH_TARGET="${CRASH_TARGET:-0.10}" \
FAIL_FAST_CHUNKS=2 \
SKIP_BUILD=1 \
bash scripts/train_sitl_plant_until_solved.sh
