#!/usr/bin/env bash
# Evaluate a zero-state Gate-2 PufferLib checkpoint from N295's measured
# official index-1 transition. Native-only and command-free.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to an FP32 PufferLib checkpoint}"
OUTPUT="${OUTPUT:?set OUTPUT to the JSON report path}"
EPISODES="${EPISODES:-512}"
SEED="${SEED:-3297}"
PERTURBED="${PERTURBED:-0}"

extra=(
    --env.start-elapsed-time-jitter 0
    --env.start-x-jitter 0 --env.start-y-jitter 0 --env.start-z-jitter 0
    --env.start-vx-jitter 0 --env.start-vy-jitter 0 --env.start-vz-jitter 0
    --env.start-roll-jitter-rad 0
    --env.start-pitch-jitter-rad 0
    --env.start-yaw-jitter-rad 0
    --env.start-wx-jitter 0 --env.start-wy-jitter 0 --env.start-wz-jitter 0
    --env.gate-position-domain-randomize 0
    --env.sitl-plant-domain-randomize 0
)
if [[ "$PERTURBED" == 1 ]]; then
    extra=(
        --env.start-elapsed-time-jitter 0.15
        --env.start-x-jitter 0.10 --env.start-y-jitter 0.10 --env.start-z-jitter 0.10
        --env.start-vx-jitter 0.20 --env.start-vy-jitter 0.10 --env.start-vz-jitter 0.10
        --env.start-roll-jitter-rad 0.015
        --env.start-pitch-jitter-rad 0.015
        --env.start-yaw-jitter-rad 0.015
        --env.start-wx-jitter 0.03 --env.start-wy-jitter 0.03 --env.start-wz-jitter 0.03
        --env.gate-position-domain-randomize 1
        --env.gate-position-domain-randomize-probability 1
        --env.gate-position-randomize-from-index 1
        --env.gate-position-jitter-x 0.30
        --env.gate-position-jitter-y 0.40
        --env.gate-position-jitter-z 0.30
        --env.sitl-plant-domain-randomize 1
        --env.sitl-rate-gain-jitter-frac 0.05
        --env.sitl-hover-thrust-jitter 0.01
        --env.sitl-rate-lag-jitter-frac 0.05
        --env.sitl-linear-drag-jitter-frac 0.05
    )
elif [[ "$PERTURBED" != 0 ]]; then
    echo "PERTURBED must be 0 or 1" >&2
    exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
python scripts/eval_drone_race_checkpoint.py "$CHECKPOINT" \
    --env-name drone_race_full_policy_six_gate_bootstrap \
    --json-path "$OUTPUT" \
    --eval-episodes "$EPISODES" \
    --require-exact-episodes \
    --horizon 64 \
    --max-rollouts 256 \
    --checkpoint-layout-precision-bytes 4 \
    --seed "$SEED" \
    --vec.total-agents 128 --vec.num-buffers 4 --vec.num-threads 8 \
    --env.num-gates 2 \
    --env.use-custom-start 1 \
    --env.start-gate-index 1 \
    --env.start-elapsed-time 3.25 \
    --env.mixed-start-curriculum 0 \
    --env.start-x 0 --env.start-y 0 --env.start-z 0 \
    --env.start-vx 4.677 --env.start-vy 0.006 --env.start-vz 0.173 \
    --env.start-qw 0.998520 --env.start-qx 0.052695 \
    --env.start-qy -0.013450 --env.start-qz 0.000931 \
    --env.start-wx 0.1226 --env.start-wy -0.0006 --env.start-wz 0 \
    --env.use-custom-gate-layout 1 \
    --env.gate0-x 0 --env.gate0-y 0 --env.gate0-z 0 \
    --env.gate1-x 14.74 --env.gate1-y 8.70 --env.gate1-z 1.095 \
    --env.gate-radius 0.75 --env.gate0-radius 0.75 --env.gate1-radius 0.75 \
    --env.observable-gate-progress 1 \
    --env.observable-gate-index-denominator 6 \
    --env.observable-gate-phase-onehot 1 \
    --env.sitl-gate-obs-sample-interval-steps 4 \
    --env.sitl-gate-obs-dropout-range-m 4.25 \
    --env.sitl-gate-obs-dropout-from-index 1 \
    --env.max-steps 1200 \
    --env.time-limit-seconds 20 \
    --env.crash-height -10 \
    --env.safety-altitude -8 \
    --env.pos-bound 45 \
    "${extra[@]}" \
    "$@"
