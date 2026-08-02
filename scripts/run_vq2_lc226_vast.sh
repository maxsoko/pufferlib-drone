#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=16
export OMP_DYNAMIC=FALSE
for profile in perception plant reset_perception reset_plant perception_plant combined_no_camera; do
    .venv/bin/python scripts/eval_vq2_lc226_raw3_interaction_isolation.py \
        --device cuda --profile "$profile"
done
