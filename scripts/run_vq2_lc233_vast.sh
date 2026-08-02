#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE

.venv/bin/python scripts/build_vq2_lc233_phase2_action_sequence.py
