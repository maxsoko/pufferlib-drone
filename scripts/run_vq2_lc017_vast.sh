#!/usr/bin/env bash
set -euo pipefail
VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed LC017 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc017_native_legal_puffer_checkpoint_screen_001}"
cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
source .venv/bin/activate
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_vq2_native_legal_puffer.py tests/test_vq2_public_phase.py
OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 \
    python scripts/eval_vq2_lc017_native_legal_puffer_checkpoints.py \
    --output "$VQ2_OUTPUT" --device cuda
