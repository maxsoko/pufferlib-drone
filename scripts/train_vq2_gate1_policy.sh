#!/usr/bin/env bash
# VQ2 N286: full-network PufferLib continuation on the measured Gate-1 domain.
# It reuses N285's native-only environment contract but removes the rejected
# single-column optimizer mask. This script cannot connect to FlightSim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/vq2_n286_gate1_policy}"
export TIMESTEPS="${TIMESTEPS:-524288}"

exec bash scripts/train_vq2_gate1_phase_adapter.sh \
    --tag vq2_n286_gate1_policy \
    --train.learning-rate 5e-5 \
    --train.min-lr-ratio 0.1 \
    --train.train-encoder-feature-start -1 \
    --train.train-encoder-feature-end -1 \
    "$@"
