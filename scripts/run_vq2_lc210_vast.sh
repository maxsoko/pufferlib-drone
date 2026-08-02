#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
.venv/bin/python scripts/eval_vq2_lc210_stacked_adapter_onpolicy2_milestone.py --device cuda "$@"
