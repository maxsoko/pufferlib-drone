#!/usr/bin/env bash
set -euo pipefail
ROOT="${REPO_ROOT:-/root/pufferlib-drone}"
cd "$ROOT"
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_ARCH=sm_86
bash build.sh drone_race
SKIP_BUILD=1 SKIP_PIP=1 FRESH_START=1 TIMESTEPS="${TIMESTEPS:-262144}" \
    bash scripts/train_drone_race_sitl_plant_local.sh
