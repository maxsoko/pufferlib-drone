#!/usr/bin/env bash
# Bounded end-to-end six-gate PPO with nominal-course anchor episodes mixed
# into late-gate position domain randomization. This script is native-only and
# never starts, resets, or connects to the official FlightSim process.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n057_r0p95_joint/drone_race_full_policy_six_gate_bootstrap/1784270090263/0000000001671168.bin}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n094_anchor_mixture/checkpoints}"
LOG_DIR="${LOG_DIR:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n094_anchor_mixture/train_logs}"
TIMESTEPS="${TIMESTEPS:-1048576}"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    echo "missing source checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 2
fi
if pgrep -af 'pufferlib.pufferl train' | grep -v "$$" >/dev/null; then
    echo "another PufferLib training process is already running" >&2
    exit 2
fi

python -m pufferlib.pufferl train drone_race_full_policy_six_gate_bootstrap \
    --load-model-path "$SOURCE_CHECKPOINT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --checkpoint-interval 1 \
    --eval-episodes 0 \
    --tag n094_six_gate_anchor_mixture \
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
    --env.gate2-radius 0.85 \
    --env.gate3-radius 0.75 \
    --env.gate4-radius 0 \
    --env.gate5-radius 0.75 \
    --env.gate-position-domain-randomize 1 \
    --env.gate-position-domain-randomize-probability 0.5 \
    --env.gate-position-randomize-from-index 4 \
    --env.gate-position-jitter-x 2 \
    --env.gate-position-jitter-y 6 \
    --env.gate-position-jitter-z 3 \
    --env.sitl-plant-domain-randomize 0 \
    --env.sitl-gate-transition-min-forward-speed 4.1875 \
    --env.w-time 0 \
    --env.w-cross-track 10 \
    --env.w-gate-crossing-error 20 \
    --env.gate-crossing-error-from-gate-index 0 \
    --train.total-timesteps "$TIMESTEPS" \
    --train.learning-rate 1e-7 \
    --train.min-lr-ratio 0.1 \
    --train.ent-coef 0 \
    --train.reward-scale 0.005 \
    "$@"
