#!/usr/bin/env bash
# Deterministically screen every admissible N209 checkpoint on both ordinary
# fixed full-course starts and the measured Gate-4 entry distribution.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

RUN_ID="${RUN_ID:-1784409692380}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/n209_measured_entry_mixture}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/checkpoints/drone_race_full_policy_six_gate_bootstrap/$RUN_ID}"
FULL_ROOT="${FULL_ROOT:-$RUN_ROOT/fixed_full_course_common128}"
ENTRY_ROOT="${ENTRY_ROOT:-$RUN_ROOT/measured_gate4_entry_common128}"
PARALLEL="${PARALLEL:-4}"
mkdir -p "$FULL_ROOT" "$ENTRY_ROOT"

common_args=(
    --env-name drone_race_full_policy_six_gate_bootstrap
    --eval-episodes 128 --require-exact-episodes
    --horizon 64 --max-rollouts 64
    --checkpoint-layout-precision-bytes 4
    --vec.total-agents 128
    --env.gate4-x 122 --env.gate4-y 5.10711 --env.gate4-z -18.24322
    --env.gate5-x 148 --env.gate5-y 5.10711 --env.gate5-z -18.24322
    --env.gate-radius 0.95
    --env.gate0-radius 0.75 --env.gate1-radius 0 --env.gate2-radius 0
    --env.gate3-radius 0.875 --env.gate4-radius 0 --env.gate5-radius 0.75
    --env.gate-position-domain-randomize 0
    --env.observable-gate-progress 1
    --env.observable-gate-index-denominator 6
    --env.observable-gate-phase-onehot 0
    --env.sitl-plant-domain-randomize 0
    --env.sitl-gate-transition-min-forward-speed 4.1875
)

screen_one() {
    local checkpoint="$1"
    local step
    step="$(basename "$checkpoint" .bin)"
    if (( 10#$step > 1048576 )); then return; fi

    if [[ ! -f "$FULL_ROOT/$step.json" ]]; then
        python scripts/eval_drone_race_checkpoint.py "$checkpoint" \
            --json-path "$FULL_ROOT/$step.json" \
            --csv-path "$FULL_ROOT/$step.csv" \
            --label "n209_fixed_full_$step" \
            "${common_args[@]}" \
            --env.use-custom-start 1 \
            --env.start-gate-index 3 \
            --env.mixed-start-curriculum 1 \
            --env.segment-start-probability 0 \
            >/dev/null
        echo "screened full $step"
    fi

    if [[ ! -f "$ENTRY_ROOT/$step.json" ]]; then
        python scripts/eval_drone_race_checkpoint.py "$checkpoint" \
            --json-path "$ENTRY_ROOT/$step.json" \
            --csv-path "$ENTRY_ROOT/$step.csv" \
            --label "n209_measured_entry_$step" \
            "${common_args[@]}" \
            --env.use-custom-start 1 \
            --env.start-gate-index 3 \
            --env.start-elapsed-time 10.85 \
            --env.start-elapsed-time-jitter 0.55 \
            --env.mixed-start-curriculum 0 \
            --env.start-x 64.3 --env.start-y -0.5 --env.start-z -6.9 \
            --env.start-x-jitter 6.0 --env.start-y-jitter 5.0 --env.start-z-jitter 2.0 \
            --env.start-vx 8.5 --env.start-vy 0.4 --env.start-vz -5.0 \
            --env.start-vx-jitter 4.0 --env.start-vy-jitter 2.0 --env.start-vz-jitter 2.5 \
            --env.start-qw 0.99529518 --env.start-qx -0.03851571 \
            --env.start-qy 0.08873983 --env.start-qz 0.00541173 \
            --env.start-roll-jitter-rad 0.22 \
            --env.start-pitch-jitter-rad 0.04 \
            --env.start-yaw-jitter-rad 0.04 \
            --env.start-wx -0.40 --env.start-wy 0.33 --env.start-wz -0.05 \
            --env.start-wx-jitter 0.25 --env.start-wy-jitter 0.30 --env.start-wz-jitter 0.10 \
            >/dev/null
        echo "screened entry $step"
    fi
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
