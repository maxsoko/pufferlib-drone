#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOVER_WEIGHTS="${HOVER_WEIGHTS:-}"
STAGE1_TIMESTEPS="${STAGE1_TIMESTEPS:-5000000}"
STAGE2_TIMESTEPS="${STAGE2_TIMESTEPS:-10000000}"
STAGE3_TIMESTEPS="${STAGE3_TIMESTEPS:-20000000}"

latest_race_checkpoint() {
    find checkpoints/drone_race -name '*.bin' -type f 2>/dev/null | sort | tail -1
}

require_checkpoint() {
    local path="$1"
    local stage="$2"
    if [ -z "$path" ] || [ ! -f "$path" ]; then
        echo "missing checkpoint after $stage" >&2
        exit 1
    fi
}

run_stage() {
    local name="$1"
    local timesteps="$2"
    local load_path="$3"
    shift 3

    echo "==> training $name for $timesteps timesteps"
    if [ -n "$load_path" ]; then
        echo "==> loading weights from $load_path"
    fi

    local load_args=()
    if [ -n "$load_path" ]; then
        load_args=(--load-model-path "$load_path")
    fi

    python -m pufferlib.pufferl train drone_race \
        "${load_args[@]}" \
        --train.total-timesteps "$timesteps" \
        "$@"
}

bash build.sh drone_race

# H1: hover-warm-started first gate. Same 23-feature observation contract as hover.
run_stage "H1 first-gate bridge" "$STAGE1_TIMESTEPS" "$HOVER_WEIGHTS" \
    --env.num-gates 1 \
    --env.gate-radius 3.0 \
    --env.gate-spacing 3.0 \
    --env.gate-lateral-amplitude 0.0 \
    --env.start-offset 1.0 \
    --env.pos-bound 30.0 \
    --env.crash-height -2.0 \
    --env.strict-missed-gate 0 \
    --env.max-steps 2400 \
    --env.time-limit-seconds 20.0 \
    --env.w-progress 60.0 \
    --env.w-gate 10.0 \
    --env.w-finish 50.0 \
    --env.w-time 0.2 \
    --env.w-ctrl 0.002 \
    --env.invalid-penalty 10.0 \
    --train.learning-rate 0.001 \
    --train.ent-coef 0.0005

STAGE1_CKPT="$(latest_race_checkpoint)"
require_checkpoint "$STAGE1_CKPT" "H1 first-gate bridge"

# R1: two gates with a mild turn.
run_stage "R1 two-gate bridge" "$STAGE2_TIMESTEPS" "$STAGE1_CKPT" \
    --env.num-gates 2 \
    --env.gate-radius 2.25 \
    --env.gate-spacing 4.0 \
    --env.gate-lateral-amplitude 0.75 \
    --env.start-offset 1.5 \
    --env.pos-bound 30.0 \
    --env.crash-height -1.0 \
    --env.strict-missed-gate 0 \
    --env.max-steps 4800 \
    --env.time-limit-seconds 40.0 \
    --env.w-progress 50.0 \
    --env.w-gate 8.0 \
    --env.w-finish 60.0 \
    --env.w-time 0.5 \
    --env.w-ctrl 0.004 \
    --env.invalid-penalty 20.0 \
    --train.learning-rate 0.001 \
    --train.ent-coef 0.0005

STAGE2_CKPT="$(latest_race_checkpoint)"
require_checkpoint "$STAGE2_CKPT" "R1 two-gate bridge"

# R2: near-full race, still slightly forgiving.
run_stage "R2 four-gate bridge" "$STAGE3_TIMESTEPS" "$STAGE2_CKPT" \
    --env.num-gates 4 \
    --env.gate-radius 1.5 \
    --env.gate-spacing 5.0 \
    --env.gate-lateral-amplitude 1.5 \
    --env.start-offset 2.0 \
    --env.pos-bound 25.0 \
    --env.crash-height -0.5 \
    --env.strict-missed-gate 1 \
    --env.max-steps 9600 \
    --env.time-limit-seconds 80.0 \
    --env.w-progress 35.0 \
    --env.w-gate 5.0 \
    --env.w-finish 75.0 \
    --env.w-time 0.75 \
    --env.w-ctrl 0.006 \
    --env.invalid-penalty 30.0 \
    --train.learning-rate 0.00075 \
    --train.ent-coef 0.00025

echo "curriculum complete; latest checkpoint: $(latest_race_checkpoint)"
