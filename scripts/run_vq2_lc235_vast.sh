#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=2
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1

.venv/bin/python scripts/build_vq2_lc235_bootstrap_sequences.py
