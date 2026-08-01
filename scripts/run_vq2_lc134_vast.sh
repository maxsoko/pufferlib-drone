#!/usr/bin/env bash
set -euo pipefail

VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc134_phase8_paired_anchor_rescue_001}"

cd "$VQ2_WORKSPACE_PATH"
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

.venv/bin/python scripts/collect_vq2_lc134_phase8_paired_anchor_rescue.py \
    --device cuda \
    --output "$VQ2_OUTPUT" \
    "$@"
