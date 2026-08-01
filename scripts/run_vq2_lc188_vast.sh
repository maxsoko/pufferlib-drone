#!/usr/bin/env bash
set -euo pipefail
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc188_phase15_adapter_onpolicy_dagger2_milestone_001}"
cd "$VQ2_WORKSPACE_PATH"
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
.venv/bin/python scripts/eval_vq2_lc188_phase15_adapter_onpolicy_dagger2_milestone.py --device cuda --output "$VQ2_OUTPUT" "$@"
