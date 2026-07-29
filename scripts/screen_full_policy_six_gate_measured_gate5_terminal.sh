#!/usr/bin/env bash
# Screen N117 children on N112's measured Gate-5 entry, with Gate 5 terminal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

RUN_ID="${RUN_ID:?set RUN_ID to the N117 checkpoint run id}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/n117_measured_gate5_terminal}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/checkpoints/drone_race_full_policy_six_gate_bootstrap/$RUN_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/fixed_measured_gate5_radius2_common128}"
PARALLEL="${PARALLEL:-4}"
mkdir -p "$OUTPUT_ROOT"

screen_one() {
    local checkpoint="$1"
    local step
    step="$(basename "$checkpoint" .bin)"
    if (( 10#$step > 524288 )); then return; fi
    if [[ -f "$OUTPUT_ROOT/$step.json" ]]; then
        echo "already screened $step"
        return
    fi
    python scripts/eval_drone_race_checkpoint.py "$checkpoint" \
        --env-name drone_race_full_policy_six_gate_bootstrap \
        --json-path "$OUTPUT_ROOT/$step.json" \
        --csv-path "$OUTPUT_ROOT/$step.csv" \
        --label "n117_measured_gate5_radius2_$step" \
        --eval-episodes 128 --require-exact-episodes \
        --horizon 64 --max-rollouts 64 \
        --checkpoint-layout-precision-bytes 4 \
        --vec.total-agents 128 \
        --env.num-gates 5 \
        --env.use-custom-start 1 \
        --env.start-gate-index 4 \
        --env.start-elapsed-time 26.7667 \
        --env.mixed-start-curriculum 0 \
        --env.start-x 95.6392746 --env.start-y 5.1144061 --env.start-z -18.3921947 \
        --env.start-vx 4.1875 --env.start-vy -0.2708830 --env.start-vz 0.0596369 \
        --env.start-qw 0.999819994 --env.start-qx -0.017756805 \
        --env.start-qy -0.006684079 --env.start-qz -0.000118691 \
        --env.start-wx -0.0266251 --env.start-wy 0.0302175 --env.start-wz 0 \
        --env.gate-radius 2 \
        --env.gate0-radius 0 --env.gate1-radius 0 --env.gate2-radius 0 \
        --env.gate3-radius 0 --env.gate4-radius 0 \
        --env.gate-position-domain-randomize 0 \
        --env.observable-gate-progress 1 \
        --env.observable-gate-index-denominator 6 \
        --env.observable-gate-phase-onehot 1 \
        --env.sitl-plant-domain-randomize 0 \
        --env.sitl-gate-transition-min-forward-speed 4.1875 \
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
