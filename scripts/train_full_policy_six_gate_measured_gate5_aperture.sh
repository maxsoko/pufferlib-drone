#!/usr/bin/env bash
# N118: fixed-target 2-4 m aperture curriculum from N112's measured Gate-4 exit.
# Native-only; never connects to official FlightSim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/n117_measured_gate5_terminal/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784317487860/0000000000294912.bin}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/n118_measured_gate5_aperture}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/train_logs}"
TIMESTEPS="${TIMESTEPS:-524288}"

if [[ ! -f "$PARENT_CHECKPOINT" ]]; then
    echo "missing N117 fixed-entry parent: $PARENT_CHECKPOINT" >&2
    exit 2
fi
if pgrep -af 'pufferlib.pufferl train' | grep -v "$$" >/dev/null; then
    echo "another PufferLib training process is already running" >&2
    exit 2
fi

python -m pufferlib.pufferl train drone_race_full_policy_six_gate_bootstrap \
    --load-model-path "$PARENT_CHECKPOINT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --checkpoint-interval 1 \
    --eval-episodes 0 \
    --tag n118_measured_gate5_fixed_aperture \
    --seed 118 \
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
    --env.gate-radius-randomize 1 \
    --env.gate-radius-min 2 --env.gate-radius-max 4 \
    --env.gate-radius-randomize-gate-index 4 \
    --env.gate0-radius 0 --env.gate1-radius 0 --env.gate2-radius 0 \
    --env.gate3-radius 0 --env.gate4-radius 0 \
    --env.gate-position-domain-randomize 0 \
    --env.observable-gate-progress 1 \
    --env.observable-gate-index-denominator 6 \
    --env.observable-gate-phase-onehot 1 \
    --env.sitl-plant-domain-randomize 0 \
    --env.sitl-gate-transition-min-forward-speed 4.1875 \
    --env.w-time 0 \
    --env.w-cross-track 10 \
    --env.w-gate-crossing-error 20 \
    --env.gate-crossing-error-from-gate-index 4 \
    --env.w-action-teacher 5 \
    --env.teacher-from-gate-index 4 \
    --env.teacher-pitch-from-gate-index 4 \
    --env.teacher-roll-from-gate-index 4 \
    --env.teacher-roll-until-gate-index 5 \
    --env.teacher-thrust-from-gate-index 4 \
    --train.total-timesteps "$TIMESTEPS" \
    --train.learning-rate 5e-4 \
    --train.min-lr-ratio 0.1 \
    --train.ent-coef 0.001 \
    --train.reward-scale 0.005 \
    --train.phase-prio-obs-index -1 \
    --train.phase-prio-scale 0 \
    "$@"
