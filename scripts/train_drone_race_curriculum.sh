#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOVER_WEIGHTS="${HOVER_WEIGHTS:-}"
H1_TIMESTEPS="${H1_TIMESTEPS:-50000000}"
R1_TIMESTEPS="${R1_TIMESTEPS:-50000000}"
R3_TIMESTEPS="${R3_TIMESTEPS:-75000000}"
R4_TIMESTEPS="${R4_TIMESTEPS:-75000000}"
H1_MIN_SUCCESS="${H1_MIN_SUCCESS:-0.95}"
H1_MAX_CRASH="${H1_MAX_CRASH:-0.05}"
R1_MIN_SUCCESS="${R1_MIN_SUCCESS:-0.90}"
R1_MAX_CRASH="${R1_MAX_CRASH:-0.10}"
R3_MIN_SUCCESS="${R3_MIN_SUCCESS:-0.85}"
R3_MAX_CRASH="${R3_MAX_CRASH:-0.10}"
R4_MIN_SUCCESS="${R4_MIN_SUCCESS:-0.85}"
R4_MAX_CRASH="${R4_MAX_CRASH:-0.10}"

latest_race_checkpoint() {
    find checkpoints/drone_race -name '*.bin' -type f 2>/dev/null | sort | tail -1
}

latest_race_run_id() {
    python - <<'PY'
from pathlib import Path

paths = sorted(Path("logs/drone_race").glob("*.json"), key=lambda p: p.stat().st_mtime)
if paths:
    print(paths[-1].stem)
PY
}

require_checkpoint() {
    local path="$1"
    local stage="$2"
    if [ -z "$path" ] || [ ! -f "$path" ]; then
        echo "missing checkpoint after $stage" >&2
        exit 1
    fi
}

validate_latest_stage() {
    local stage="$1"
    local min_success="$2"
    local max_crash="$3"
    local run_id
    run_id="$(latest_race_run_id)"
    if [ -z "$run_id" ]; then
        echo "missing logs after $stage" >&2
        exit 1
    fi

    python - "$run_id" "$stage" "$min_success" "$max_crash" <<'PY'
import json
import sys
from pathlib import Path

run_id, stage, min_success, max_crash = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
path = Path("logs/drone_race") / f"{run_id}.json"
data = json.loads(path.read_text())
metrics = data.get("metrics", {})

def last(key):
    vals = metrics.get(key)
    if not vals:
        raise SystemExit(f"missing metric {key} in {path}")
    return float(vals[-1])

success = last("env/success_rate")
crash = last("env/crash")
gates = last("env/gates_passed")
print(f"{stage}: run={run_id} success={success:.4f} crash={crash:.4f} gates_passed={gates:.4f}")
if success < min_success or crash > max_crash:
    raise SystemExit(
        f"{stage} failed promotion thresholds: "
        f"success >= {min_success:.3f}, crash <= {max_crash:.3f}"
    )
PY
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
run_stage "H1 first-gate bridge" "$H1_TIMESTEPS" "$HOVER_WEIGHTS" \
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

validate_latest_stage "H1 first-gate bridge" "$H1_MIN_SUCCESS" "$H1_MAX_CRASH"
STAGE1_CKPT="$(latest_race_checkpoint)"
require_checkpoint "$STAGE1_CKPT" "H1 first-gate bridge"

# R1: two gates with a mild turn.
run_stage "R1 two-gate bridge" "$R1_TIMESTEPS" "$STAGE1_CKPT" \
    --env.num-gates 2 \
    --env.gate-radius 2.25 \
    --env.gate-spacing 4.0 \
    --env.gate-lateral-amplitude 0.75 \
    --env.start-offset 1.5 \
    --env.pos-bound 30.0 \
    --env.crash-height -1.5 \
    --env.strict-missed-gate 0 \
    --env.max-steps 4800 \
    --env.time-limit-seconds 40.0 \
    --env.w-progress 45.0 \
    --env.w-gate 8.0 \
    --env.w-finish 70.0 \
    --env.w-time 0.25 \
    --env.w-ctrl 0.008 \
    --env.invalid-penalty 30.0 \
    --train.learning-rate 0.0005 \
    --train.ent-coef 0.00025

validate_latest_stage "R1 two-gate bridge" "$R1_MIN_SUCCESS" "$R1_MAX_CRASH"
STAGE2_CKPT="$(latest_race_checkpoint)"
require_checkpoint "$STAGE2_CKPT" "R1 two-gate bridge"

# R3: three-gate bridge. Vast runs showed direct four-gate promotion regresses.
run_stage "R3 three-gate bridge" "$R3_TIMESTEPS" "$STAGE2_CKPT" \
    --env.num-gates 3 \
    --env.gate-radius 1.9 \
    --env.gate-spacing 4.5 \
    --env.gate-lateral-amplitude 1.0 \
    --env.start-offset 1.75 \
    --env.pos-bound 30.0 \
    --env.crash-height -1.0 \
    --env.strict-missed-gate 0 \
    --env.max-steps 7200 \
    --env.time-limit-seconds 60.0 \
    --env.w-progress 40.0 \
    --env.w-gate 7.0 \
    --env.w-finish 80.0 \
    --env.w-time 0.3 \
    --env.w-ctrl 0.008 \
    --env.invalid-penalty 35.0 \
    --train.learning-rate 0.0005 \
    --train.ent-coef 0.00025

validate_latest_stage "R3 three-gate bridge" "$R3_MIN_SUCCESS" "$R3_MAX_CRASH"
STAGE3_CKPT="$(latest_race_checkpoint)"
require_checkpoint "$STAGE3_CKPT" "R3 three-gate bridge"

# R4: near-full race, still slightly forgiving.
run_stage "R4 four-gate bridge" "$R4_TIMESTEPS" "$STAGE3_CKPT" \
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

validate_latest_stage "R4 four-gate bridge" "$R4_MIN_SUCCESS" "$R4_MAX_CRASH"
echo "curriculum complete; latest checkpoint: $(latest_race_checkpoint)"
