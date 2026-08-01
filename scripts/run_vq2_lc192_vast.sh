#!/usr/bin/env bash
set -euo pipefail
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc192_phase17_failure_state_dagger_001}"
cd "$VQ2_WORKSPACE_PATH"
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
.venv/bin/python scripts/collect_vq2_lc192_phase17_failure_state_dagger.py --device cuda --output "$VQ2_OUTPUT" "$@"
