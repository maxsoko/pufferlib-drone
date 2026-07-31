#!/usr/bin/env bash
# Source-locked Vast VG069 phase-independent early-stop replay.
set -euo pipefail
VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG069 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg069_phase_independent_early_stop_001}"
cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
source .venv/bin/activate
if [ -f "${VQ2_OUTPUT}_state.json" ]; then
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python scripts/train_vq2_vg069_phase_independent_early_stop.py --output "$VQ2_OUTPUT" --device cuda --resume
    exit $?
fi
test "$(sha256sum "$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg063_horizon_corrected_intervention_features_001/features.bin" | cut -d" " -f1)" = \
    ccfc07e578e1f16900b9d020c5c9063676058c518b8524bf5452738566181580
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_train_vq2_vg069_phase_independent_early_stop.py \
    tests/test_train_vq2_vg068_indexed_mlp_intervention_features.py \
    tests/test_vq2_recurrent_phase_residual.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python scripts/train_vq2_vg069_phase_independent_early_stop.py --output "$VQ2_OUTPUT" --device cuda
