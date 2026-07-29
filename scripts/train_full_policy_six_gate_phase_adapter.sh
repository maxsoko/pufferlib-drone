#!/usr/bin/env bash
# N104: one bounded full-start, six-gate PPO run that adds a legal six-phase
# one-hot observation and trains only the Gate-4/5/6 encoder columns. Gates
# 1-3 and every shared/recurrent/decoder parameter remain byte-exact.
# Native-only: this script never connects to or resets official FlightSim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n096_raw_anchor_mixture/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784286271692/0000000000491520.bin}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n104_phase_adapter}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$RUN_ROOT/source/n096_reserved_zero.bin}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/train_logs}"
TIMESTEPS="${TIMESTEPS:-1048576}"

if [[ ! -f "$PARENT_CHECKPOINT" ]]; then
    echo "missing parent checkpoint: $PARENT_CHECKPOINT" >&2
    exit 2
fi
if pgrep -af 'pufferlib.pufferl train' | grep -v "$$" >/dev/null; then
    echo "another PufferLib training process is already running" >&2
    exit 2
fi

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    python scripts/zero_policy_encoder_feature.py \
        "$PARENT_CHECKPOINT" "$SOURCE_CHECKPOINT" \
        --input-dim 32 \
        --layout-precision-bytes 4 \
        --feature-index 24 \
        --feature-index 25 \
        --feature-index 26 \
        --feature-index 27 \
        --feature-index 28 \
        --feature-index 29 \
        --feature-index 30 \
        --feature-index 31
fi

python -m pufferlib.pufferl train drone_race_full_policy_six_gate_bootstrap \
    --load-model-path "$SOURCE_CHECKPOINT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --checkpoint-interval 1 \
    --eval-episodes 0 \
    --tag n104_six_gate_phase_adapter \
    --seed 94 \
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
    --env.gate4-x 122 \
    --env.gate4-y 5.10711 \
    --env.gate4-z -18.24322 \
    --env.gate5-x 148 \
    --env.gate5-y 5.10711 \
    --env.gate5-z -18.24322 \
    --env.gate-radius 0.95 \
    --env.gate0-radius 0.75 \
    --env.gate1-radius 0 \
    --env.gate2-radius 0 \
    --env.gate3-radius 0.875 \
    --env.gate4-radius 0 \
    --env.gate5-radius 0.75 \
    --env.gate-position-domain-randomize 1 \
    --env.gate-position-domain-randomize-probability 0.5 \
    --env.gate-position-randomize-from-index 4 \
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
    --env.gate-crossing-error-from-gate-index 0 \
    --train.total-timesteps "$TIMESTEPS" \
    --train.learning-rate 1e-3 \
    --train.min-lr-ratio 0.1 \
    --train.ent-coef 0 \
    --train.reward-scale 0.005 \
    --train.phase-prio-obs-index -1 \
    --train.phase-prio-scale 0 \
    --train.train-encoder-feature-start 27 \
    --train.train-encoder-feature-end 29 \
    "$@"
