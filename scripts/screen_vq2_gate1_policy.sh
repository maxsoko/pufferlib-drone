#!/usr/bin/env bash
# Screen N286 checkpoints on the exact passive VQ2 Gate-1 measurement.
# Native-only: this script never opens simulator UDP ports or sends commands.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

RUN_ID="${RUN_ID:-1785123191717}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/vq2_n286_gate1_policy}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/checkpoints/drone_race_full_policy_six_gate_bootstrap/$RUN_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/fixed_measured_gate1_common128}"
PARALLEL="${PARALLEL:-2}"
mkdir -p "$OUTPUT_ROOT"

screen_one() {
    local checkpoint="$1"
    local step
    step="$(basename "$checkpoint" .bin)"
    if [[ -f "$OUTPUT_ROOT/$step.json" ]]; then
        echo "already screened $step"
        return
    fi
    python scripts/eval_drone_race_checkpoint.py "$checkpoint" \
        --env-name drone_race_full_policy_six_gate_bootstrap \
        --json-path "$OUTPUT_ROOT/$step.json" \
        --csv-path "$OUTPUT_ROOT/$step.csv" \
        --label "vq2_n286_fixed_gate1_$step" \
        --eval-episodes 128 \
        --require-exact-episodes \
        --horizon 64 \
        --max-rollouts 256 \
        --checkpoint-layout-precision-bytes 4 \
        --seed 1286 \
        --vec.total-agents 128 \
        --vec.num-buffers 4 \
        --vec.num-threads 8 \
        --env.num-gates 1 \
        --env.use-custom-start 1 \
        --env.start-gate-index 0 \
        --env.start-elapsed-time 0 \
        --env.mixed-start-curriculum 0 \
        --env.start-x 0 \
        --env.start-y 0 \
        --env.start-z 0 \
        --env.start-vx 0 \
        --env.start-vy 0 \
        --env.start-vz 0 \
        --env.start-qw 1 \
        --env.start-qx 0 \
        --env.start-qy 0 \
        --env.start-qz 0 \
        --env.start-wx 0 \
        --env.start-wy 0 \
        --env.start-wz 0 \
        --env.use-custom-gate-layout 1 \
        --env.gate0-x 10.78 \
        --env.gate0-y 0.084 \
        --env.gate0-z 0.56 \
        --env.gate-radius 0.75 \
        --env.gate0-radius 0.75 \
        --env.gate-position-domain-randomize 0 \
        --env.observable-gate-progress 1 \
        --env.observable-gate-index-denominator 6 \
        --env.observable-gate-phase-onehot 1 \
        --env.sitl-plant-domain-randomize 0 \
        --env.sitl-gate-obs-sample-interval-steps 4 \
        --env.sitl-gate-obs-dropout-range-m 4.25 \
        --env.sitl-gate-obs-dropout-from-index 0 \
        --env.max-steps 900 \
        --env.time-limit-seconds 15 \
        --env.crash-height -10 \
        --env.safety-altitude -8 \
        --env.pos-bound 40 \
        >/dev/null
    echo "screened $step"
}

active=0
for checkpoint in "$CHECKPOINT_ROOT"/*.bin; do
    screen_one "$checkpoint" &
    active=$((active + 1))
    if (( active >= PARALLEL )); then
        wait -n
        active=$((active - 1))
    fi
done
wait
