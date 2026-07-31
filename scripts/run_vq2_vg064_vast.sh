#!/usr/bin/env bash
# Source-locked Vast VG064 intervention-feature distillation.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG064 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg064_indexed_intervention_ridge_001}"
VQ2_DATASET="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg063_horizon_corrected_intervention_features_001"
VQ2_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg063_horizon_corrected_intervention_admission_2026-07-31.json"

run_fit() {
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/train_vq2_vg064_indexed_intervention_ridge.py \
            --output "$VQ2_OUTPUT" --device cuda "$@"
}

cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test -x .venv/bin/python
source .venv/bin/activate
if [ -f "${VQ2_OUTPUT}_state.json" ]; then
    run_fit --resume
    exit $?
fi
for command_name in git nvidia-smi python; do command -v "$command_name" >/dev/null; done
test "$(sha256sum "$VQ2_DATASET/report.json" | cut -d" " -f1)" = \
    d74f16bdd6d9da17a4fffedf52e53cb34173940d0e6943b755c3c683bd1a583d
test "$(sha256sum "$VQ2_DATASET/features.bin" | cut -d" " -f1)" = \
    ccfc07e578e1f16900b9d020c5c9063676058c518b8524bf5452738566181580
test "$(sha256sum "$VQ2_ADMISSION" | cut -d" " -f1)" = \
    47c1e107dbf283d49f4b44e968d139a8743a9d3aa194777a23e175099f02e3cc
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_train_vq2_vg064_indexed_intervention_ridge.py \
    tests/test_vq2_recurrent_phase_residual.py
run_fit
