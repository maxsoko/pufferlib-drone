#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1

.venv/bin/python scripts/collect_vq2_lc230_numpy_batch1_phase2_rescue.py --device cuda
