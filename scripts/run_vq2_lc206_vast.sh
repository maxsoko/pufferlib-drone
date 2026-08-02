#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=FALSE
.venv/bin/python scripts/train_vq2_lc206_stacked_adapter_onpolicy_dagger.py --device cuda "$@"
