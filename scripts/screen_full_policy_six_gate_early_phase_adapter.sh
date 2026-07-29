#!/usr/bin/env bash
# Screen all admissible N107 children on the preregistered fixed six-gate
# all-radius-0.75 common-128 target. Native-only; never connects to FlightSim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

RUN_ID="${RUN_ID:-1784299758401}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n107_early_phase_adapter}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$RUN_ROOT/checkpoints/drone_race_full_policy_six_gate_bootstrap/$RUN_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$RUN_ROOT/fixed_all_radius0p75_common128}"
PARALLEL="${PARALLEL:-2}"
LABEL_PREFIX="${LABEL_PREFIX:-n107}"
mkdir -p "$OUTPUT_ROOT"

screen_one() {
    local checkpoint="$1"
    local step
    step="$(basename "$checkpoint" .bin)"
    if (( 10#$step > 1048576 )); then
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
        --label "${LABEL_PREFIX}_$step" \
        --eval-episodes 128 \
        --require-exact-episodes \
        --horizon 64 \
        --max-rollouts 64 \
        --checkpoint-layout-precision-bytes 4 \
        --vec.total-agents 128 \
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
        --env.gate4-x 122 \
        --env.gate4-y 5.10711 \
        --env.gate4-z -18.24322 \
        --env.gate5-x 148 \
        --env.gate5-y 5.10711 \
        --env.gate5-z -18.24322 \
        --env.gate-radius 0.75 \
        --env.gate0-radius 0 \
        --env.gate1-radius 0 \
        --env.gate2-radius 0 \
        --env.gate3-radius 0 \
        --env.gate4-radius 0 \
        --env.gate5-radius 0 \
        --env.gate-position-domain-randomize 0 \
        --env.observable-gate-progress 1 \
        --env.observable-gate-index-denominator 6 \
        --env.observable-gate-phase-onehot 1 \
        --env.sitl-plant-domain-randomize 0 \
        --env.sitl-gate-transition-min-forward-speed 4.1875 \
        --env.sitl-gate-obs-dropout-range-m 4.25 \
        --env.sitl-gate-obs-dropout-from-index 1 \
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
