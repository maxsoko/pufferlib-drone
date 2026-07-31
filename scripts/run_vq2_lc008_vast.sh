#!/usr/bin/env bash
# Source-locked Vast LC008 long-course intervention feature collection.
set -euo pipefail
VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed LC008 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc008_long_course_intervention_features_001}"
cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
source .venv/bin/activate
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase_residual.py \
    tests/test_collect_vq2_vg062_warmed_teacher_intervention_features.py
OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 \
    python scripts/collect_vq2_lc008_long_course_intervention_features.py \
    --output "$VQ2_OUTPUT" --device cuda
