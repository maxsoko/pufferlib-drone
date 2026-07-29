#!/usr/bin/env bash
# Screen N115B children on the fixed Gates-5-through-6 segment. Native-only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

RUN_ID="${RUN_ID:?set RUN_ID to the N115B checkpoint run id}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/n115_backward_tail_gate5}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/checkpoints/drone_race_full_policy_six_gate_bootstrap/$RUN_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/fixed_gate5_radius2_common128}"
PARALLEL="${PARALLEL:-4}"
mkdir -p "$OUTPUT_ROOT"

screen_one() {
    local checkpoint="$1"
    local step
    step="$(basename "$checkpoint" .bin)"
    if (( 10#$step > 524288 )); then
        return
    fi
    if [[ -f "$OUTPUT_ROOT/$step.json" ]]; then
        echo "already screened $step"
        return
    fi
    python scripts/eval_drone_race_checkpoint.py "$checkpoint" \
        --env-name drone_race_full_policy_six_gate_bootstrap \
        --json-path "$OUTPUT_ROOT/$step.json" \
        --csv-path "$OUTPUT_ROOT/$step.csv" \
        --label "n115b_gate5_radius2_$step" \
        --eval-episodes 128 --require-exact-episodes \
        --horizon 64 --max-rollouts 64 \
        --checkpoint-layout-precision-bytes 4 \
        --vec.total-agents 128 \
        --env.use-custom-start 1 \
        --env.start-gate-index 4 \
        --env.start-elapsed-time 12 \
        --env.mixed-start-curriculum 0 \
        --env.start-x 95.6318 --env.start-y 5.10711 --env.start-z -18.24322 \
        --env.start-vx 8 --env.start-vy -2.459 --env.start-vz 0.0738 \
        --env.start-qw 1 --env.start-qx 0 --env.start-qy 0 --env.start-qz 0 \
        --env.start-wx 0 --env.start-wy 0 --env.start-wz 0 \
        --env.gate-radius 2 \
        --env.gate0-radius 0 --env.gate1-radius 0 --env.gate2-radius 0 \
        --env.gate3-radius 0 --env.gate4-radius 0 --env.gate5-radius 0 \
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
