#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
.venv/bin/python scripts/eval_vq2_lc197_phase17_exact_context_bracket.py --device cuda
