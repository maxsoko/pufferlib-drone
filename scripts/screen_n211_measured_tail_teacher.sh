#!/usr/bin/env bash
# Screen every admissible N211 checkpoint on a common measured Gate-4 cohort.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

RUN_ID="${RUN_ID:?set RUN_ID to the completed N211 checkpoint run id}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/n211_measured_tail_teacher}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/checkpoints/drone_race_full_policy_six_gate_bootstrap/$RUN_ID}"
ENTRY_ROOT="${ENTRY_ROOT:-$RUN_ROOT/measured_gate4_entry_common256}"
PARALLEL="${PARALLEL:-4}"
EPISODES="${EPISODES:-256}"
STEP="${STEP:-}"
EXTRA_ARGS=("$@")
mkdir -p "$ENTRY_ROOT"

screen_one() {
    local checkpoint="$1"
    local step
    step="$(basename "$checkpoint" .bin)"
    if [[ -n "$STEP" && "$step" != "$STEP" ]]; then return; fi
    if [[ "$step" =~ ^[0-9]+$ ]] && (( 10#$step > 524288 )); then return; fi
    if [[ -f "$ENTRY_ROOT/$step.json" ]]; then return; fi
    python scripts/eval_drone_race_checkpoint.py "$checkpoint" \
        --json-path "$ENTRY_ROOT/$step.json" \
        --csv-path "$ENTRY_ROOT/$step.csv" \
        --label "n211_measured_entry_$step" \
        --env-name drone_race_full_policy_six_gate_bootstrap \
        --eval-episodes "$EPISODES" --require-exact-episodes \
        --horizon 64 --max-rollouts 64 \
        --checkpoint-layout-precision-bytes 4 \
        --vec.total-agents "$EPISODES" \
        --env.gate4-x 122 --env.gate4-y 5.10711 --env.gate4-z -18.24322 \
        --env.gate5-x 148 --env.gate5-y 5.10711 --env.gate5-z -18.24322 \
        --env.gate-radius 0.95 \
        --env.gate0-radius 0.75 --env.gate1-radius 0 --env.gate2-radius 0 \
        --env.gate3-radius 0.875 --env.gate4-radius 0 --env.gate5-radius 0.75 \
        --env.gate-position-domain-randomize 0 \
        --env.observable-gate-progress 1 \
        --env.observable-gate-index-denominator 6 \
        --env.observable-gate-phase-onehot 0 \
        --env.sitl-plant-domain-randomize 0 \
        --env.sitl-gate-transition-min-forward-speed 4.1875 \
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
        "${EXTRA_ARGS[@]}" \
        >/dev/null
    echo "screened entry $step"
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
