#!/usr/bin/env bash
# Retry native drone_race_sitl_plant training until a chunk completes without heap abort.
# Only run ONE instance at a time (concurrent pufferl processes corrupt WSL heap).
set -euo pipefail

ROOT="${REPO_ROOT:-$HOME/pufferlib-drone}"
cd "$ROOT"
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_ARCH=sm_86

TIMESTEPS="${TIMESTEPS:-131072}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
ENV_NAME="${ENV_NAME:-drone_race_sitl_plant}"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    echo "==> native train attempt ${attempt}/${MAX_ATTEMPTS} (${TIMESTEPS} steps)"
    if FRESH_START=1 SKIP_BUILD=1 SKIP_PIP=1 TIMESTEPS="$TIMESTEPS" ENV_NAME="$ENV_NAME" \
        bash scripts/train_drone_race_sitl_plant_local.sh; then
        echo "SUCCESS on attempt ${attempt}"
        exit 0
    fi
    echo "attempt ${attempt} failed; sleeping before retry" >&2
    sleep 3
done

echo "all ${MAX_ATTEMPTS} attempts failed" >&2
exit 1
