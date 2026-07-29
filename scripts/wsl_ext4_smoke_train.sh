#!/usr/bin/env bash
set -euo pipefail
ROOT="${REPO_ROOT:-$HOME/pufferlib-drone}"
cd "$ROOT"
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_ARCH=sm_86
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    bash build.sh drone_race
fi
LOAD_PATH="${LOAD_PATH:-}" TIMESTEPS="${TIMESTEPS:-131072}" ENV_NAME="${ENV_NAME:-drone_race_sitl_plant}" \
    bash scripts/train_drone_race_sitl_plant_local.sh
