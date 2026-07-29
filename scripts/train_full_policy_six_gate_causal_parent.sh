#!/usr/bin/env bash
# N109: one full-network PPO rung from N107's causal early-phase checkpoint,
# using N108's exact anchored six-gate aperture/geometry mixture. Native-only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n107_early_phase_adapter/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784299758401/0000000000327680.bin}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n109_causal_parent}"
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

python -m pufferlib.pufferl train drone_race_full_policy_six_gate_bootstrap \
    --load-model-path "$PARENT_CHECKPOINT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --checkpoint-interval 1 \
    --eval-episodes 0 \
    --tag n109_six_gate_causal_parent \
    --seed 97 \
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
    --env.gate-radius-randomize 0 \
    --env.gate-radius-profile-mix 1 \
    --env.gate-radius-profile-mix-probability 0.5 \
    --env.gate-radius-profile-mix-target 0.75 \
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
    --env.sitl-gate-obs-dropout-range-m 4.25 \
    --env.sitl-gate-obs-dropout-from-index 1 \
    --env.w-time 0 \
    --env.w-cross-track 10 \
    --env.w-gate-crossing-error 20 \
    --env.gate-crossing-error-from-gate-index 0 \
    --train.total-timesteps "$TIMESTEPS" \
    --train.learning-rate 1e-6 \
    --train.min-lr-ratio 0.1 \
    --train.ent-coef 0 \
    --train.reward-scale 0.005 \
    --train.phase-prio-obs-index -1 \
    --train.phase-prio-scale 0 \
    --train.train-encoder-feature-start -1 \
    --train.train-encoder-feature-end -1 \
    "$@"
