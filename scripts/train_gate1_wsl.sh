#!/usr/bin/env bash
# One-shot: build + train drone_race_sitl_plant_gate1 on WSL ext4.
set -euo pipefail

ROOT="${REPO_ROOT:-$HOME/pufferlib-drone}"
cd "$ROOT"
source .venv/bin/activate

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_ARCH="${NVCC_ARCH:-sm_86}"

echo "==> build drone_race (custom gate layout)"
bash build.sh drone_race

echo "==> native regressions"
bash scripts/test_drone_race_native_regressions.sh

# Use drone_race_sitl_plant_gate1_hover after fast config failed (hover-stabilize curriculum).
export ENV_NAME="${ENV_NAME:-drone_race_sitl_plant_gate1_hover}"
export FRESH_START=1
export CHUNK_STEPS="${CHUNK_STEPS:-1000000}"
export MAX_CHUNKS="${MAX_CHUNKS:-10}"
export SKIP_BUILD=1

echo "==> train until solved: $ENV_NAME"
bash scripts/train_sitl_plant_until_solved.sh
