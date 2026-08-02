#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE

.venv/bin/python scripts/train_vq2_lc231_phase2_full_residual.py --device cuda
