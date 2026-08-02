#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
for profile in control reset camera_jitter camera_dropout edge_dropout rolling_shutter plant_gain_hover plant_lag_drag combined; do
    .venv/bin/python scripts/eval_vq2_lc223_raw2_perturbation_isolation.py \
        --device cuda --profile "$profile"
done
