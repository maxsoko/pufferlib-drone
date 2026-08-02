#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
for profile in control reset camera_dropout edge_dropout rolling_shutter reset_gain_hover reset_lag_drag; do
    .venv/bin/python scripts/eval_vq2_lc227_raw3_factor_isolation.py \
        --device cuda --profile "$profile"
done
