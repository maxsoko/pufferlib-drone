#!/usr/bin/env bash
set -euo pipefail
VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed LC046 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc046_phase6_student_head_001}"
cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
source .venv/bin/activate
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_train_vq2_lc013_index1_student_dagger_head.py \
    tests/test_vq2_recurrent_phase_residual.py
OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 \
    python scripts/train_vq2_lc046_phase6_student_head.py \
    --output "$VQ2_OUTPUT" --device cuda
