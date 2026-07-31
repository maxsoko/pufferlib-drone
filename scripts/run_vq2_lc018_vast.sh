#!/usr/bin/env bash
set -euo pipefail
VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed LC018 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc018_native_gate1_lowstd_ppo_001}"
cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
source .venv/bin/activate
scripts/test_drone_race_vision_native_regressions.sh
OMP_NUM_THREADS=128 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 \
    python scripts/train_vq2_lc018_native_gate1_lowstd_ppo.py \
    --output "$VQ2_OUTPUT"
