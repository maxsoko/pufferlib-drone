#!/usr/bin/env bash
set -euo pipefail
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc180_phase15_recurrent_adapter_early_weighted_001}"
cd "$VQ2_WORKSPACE_PATH"
export OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python scripts/train_vq2_lc180_phase15_recurrent_adapter_early_weighted.py --device cuda --output "$VQ2_OUTPUT" "$@"
