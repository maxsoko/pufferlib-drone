#!/usr/bin/env bash
# Fast two-stage native curriculum for the v3385 gate-2 hybrid policy.
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

echo "==> stage 1: safe short-gate acquisition"
REPO_ROOT="$ROOT" \
ENV_NAME=drone_race_sitl_plant_gate1_hover \
CHUNK_STEPS="${EASY_CHUNK_STEPS:-2000000}" \
MAX_CHUNKS="${EASY_MAX_CHUNKS:-8}" \
EVAL_EPISODES="${EVAL_EPISODES:-4096}" \
SUCCESS_TARGET="${EASY_SUCCESS_TARGET:-0.90}" \
CRASH_TARGET="${EASY_CRASH_TARGET:-0.10}" \
FAIL_FAST_CHUNKS=2 \
SKIP_BUILD=1 \
bash scripts/train_sitl_plant_until_solved.sh

easy_checkpoint="$(<logs/drone_race_sitl_plant_gate1_hover_solved_checkpoint.txt)"
echo "==> stage 2: long descending gate-2 geometry from $easy_checkpoint"
REPO_ROOT="$ROOT" \
ENV_NAME=drone_race_sitl_plant_gate2_fast \
LOAD_PATH="$easy_checkpoint" \
FRESH_START=1 \
CHUNK_STEPS="${HARD_CHUNK_STEPS:-3000000}" \
MAX_CHUNKS="${HARD_MAX_CHUNKS:-12}" \
EVAL_EPISODES="${EVAL_EPISODES:-4096}" \
SUCCESS_TARGET="${HARD_SUCCESS_TARGET:-0.80}" \
CRASH_TARGET="${HARD_CRASH_TARGET:-0.15}" \
FAIL_FAST_CHUNKS=2 \
SKIP_BUILD=1 \
bash scripts/train_sitl_plant_until_solved.sh

checkpoint="$(<logs/drone_race_sitl_plant_gate2_fast_solved_checkpoint.txt)"
echo "HYBRID_POLICY_READY=$checkpoint"
echo "PowerShell validation:"
echo ".\\scripts\\run_windows_course_fsm.ps1 -PolicyCheckpoint '$checkpoint' -PolicyActivateGateIndex 1 -SmokeDuration 30 -TargetGateCount 2 -Tag hybrid_gate2"
