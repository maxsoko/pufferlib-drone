#!/usr/bin/env bash
# Source-locked Vast VG065 group-stratified intervention-feature fit.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG065 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg065_group_stratified_intervention_ridge_001}"
VQ2_FAILURE="$VQ2_WORKSPACE_PATH/docs/vq2_vg064_validation_split_infrastructure_rejection_2026-07-31.json"

run_fit() {
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/train_vq2_vg065_group_stratified_intervention_ridge.py \
            --output "$VQ2_OUTPUT" --device cuda "$@"
}

cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test -x .venv/bin/python
source .venv/bin/activate
if [ -f "${VQ2_OUTPUT}_state.json" ]; then run_fit --resume; exit $?; fi
test "$(sha256sum "$VQ2_FAILURE" | cut -d" " -f1)" = \
    a38ca55065a822ec8f4f2e480be504c512061d1df8dab7164bd2d34c83d648da
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_train_vq2_vg064_indexed_intervention_ridge.py \
    tests/test_train_vq2_vg065_group_stratified_intervention_ridge.py \
    tests/test_vq2_recurrent_phase_residual.py
run_fit
