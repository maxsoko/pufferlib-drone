#!/usr/bin/env bash
# Source-locked Vast VG070 teacher-free count-5 nonlinear scale bracket.
set -euo pipefail
VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG070 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg070_nonlinear_residual_count5_scale_bracket_001}"
cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
source .venv/bin/activate
if [ -f "$VQ2_OUTPUT/state.json" ]; then
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python scripts/eval_vq2_vg070_nonlinear_residual_count5_scale_bracket.py --output "$VQ2_OUTPUT" --device cuda --resume
    exit $?
fi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_eval_vq2_vg070_nonlinear_residual_count5_scale_bracket.py \
    tests/test_vq2_recurrent_phase_residual.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_vq2_staged_count5_diagnostic.py -k 'not real_manifests'
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python scripts/eval_vq2_vg070_nonlinear_residual_count5_scale_bracket.py --output "$VQ2_OUTPUT" --device cuda
