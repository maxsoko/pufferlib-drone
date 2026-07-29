#!/usr/bin/env bash
# N115A: first stage of a backward tail curriculum. Train the final gate alone;
# never connects to or controls official FlightSim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n104_phase_adapter/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784295732298/0000000001048576.bin}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/n115_backward_tail_gate6}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/train_logs}"
TIMESTEPS="${TIMESTEPS:-524288}"

if [[ ! -f "$PARENT_CHECKPOINT" ]]; then
    echo "missing N104 tail parent: $PARENT_CHECKPOINT" >&2
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
    --tag n115a_backward_tail_gate6 \
    --seed 115 \
    --env.use-custom-start 1 \
    --env.start-gate-index 5 \
    --env.start-elapsed-time 15 \
    --env.mixed-start-curriculum 0 \
    --env.start-x 122 --env.start-y -3 --env.start-z -18 \
    --env.start-vx 8 --env.start-vy 2.153846 --env.start-vz 0.307692 \
    --env.start-qw 1 --env.start-qx 0 --env.start-qy 0 --env.start-qz 0 \
    --env.start-wx 0 --env.start-wy 0 --env.start-wz 0 \
    --env.gate-radius 2 \
    --env.gate0-radius 0 --env.gate1-radius 0 --env.gate2-radius 0 \
    --env.gate3-radius 0 --env.gate4-radius 0 --env.gate5-radius 0 \
    --env.gate-position-domain-randomize 1 \
    --env.gate-position-domain-randomize-probability 1 \
    --env.gate-position-randomize-from-index 5 \
    --env.gate-position-jitter-x 2 \
    --env.gate-position-jitter-y 6 \
    --env.gate-position-jitter-z 3 \
    --env.observable-gate-progress 1 \
    --env.observable-gate-index-denominator 6 \
    --env.observable-gate-phase-onehot 1 \
    --env.sitl-plant-domain-randomize 0 \
    --env.sitl-gate-transition-min-forward-speed 4.1875 \
    --env.w-time 0 \
    --env.w-cross-track 10 \
    --env.w-gate-crossing-error 20 \
    --env.gate-crossing-error-from-gate-index 5 \
    --env.w-action-teacher 5 \
    --env.teacher-from-gate-index 5 \
    --env.teacher-pitch-from-gate-index 5 \
    --env.teacher-roll-from-gate-index 5 \
    --env.teacher-roll-until-gate-index 6 \
    --env.teacher-thrust-from-gate-index 5 \
    --train.total-timesteps "$TIMESTEPS" \
    --train.learning-rate 1e-3 \
    --train.min-lr-ratio 0.1 \
    --train.ent-coef 0.001 \
    --train.reward-scale 0.005 \
    --train.phase-prio-obs-index -1 \
    --train.phase-prio-scale 0 \
    "$@"
