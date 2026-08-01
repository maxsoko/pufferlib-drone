#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
.venv/bin/python scripts/collect_vq2_lc193_phase16_17_rescue_dagger.py --device cuda
