#!/usr/bin/env bash
# VQ2 N287+: one native-only PufferLib continuation stage on the exact
# passively measured Gate-1 center. GATE_RADIUS controls curriculum difficulty.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n104_phase_adapter/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784295732298/0000000001048576.bin}"
GATE_RADIUS="${GATE_RADIUS:-1.60}"
STAGE_NAME="${STAGE_NAME:-vq2_n287_gate1_radius1p60}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/$STAGE_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/train_logs}"
TIMESTEPS="${TIMESTEPS:-262144}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

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
    --tag "$STAGE_NAME" \
    --seed 287 \
    --vec.total-agents 128 \
    --vec.num-buffers 4 \
    --vec.num-threads 8 \
    --env.num-gates 1 \
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
    --env.start-x-jitter 0 \
    --env.start-y-jitter 0 \
    --env.start-z-jitter 0 \
    --env.start-vx-jitter 0 \
    --env.start-vy-jitter 0 \
    --env.start-vz-jitter 0 \
    --env.start-qw 1 \
    --env.start-qx 0 \
    --env.start-qy 0 \
    --env.start-qz 0 \
    --env.start-roll-jitter-rad 0 \
    --env.start-pitch-jitter-rad 0 \
    --env.start-yaw-jitter-rad 0 \
    --env.start-wx 0 \
    --env.start-wy 0 \
    --env.start-wz 0 \
    --env.start-wx-jitter 0 \
    --env.start-wy-jitter 0 \
    --env.start-wz-jitter 0 \
    --env.use-custom-gate-layout 1 \
    --env.gate0-x 10.78 \
    --env.gate0-y 0.084 \
    --env.gate0-z 0.56 \
    --env.gate-radius "$GATE_RADIUS" \
    --env.gate0-radius "$GATE_RADIUS" \
    --env.gate-position-domain-randomize 0 \
    --env.observable-gate-progress 1 \
    --env.observable-gate-index-denominator 6 \
    --env.observable-gate-phase-onehot 1 \
    --env.sitl-plant-domain-randomize 0 \
    --env.sitl-gate-obs-sample-interval-steps 4 \
    --env.sitl-gate-obs-dropout-range-m 4.25 \
    --env.sitl-gate-obs-dropout-from-index 0 \
    --env.max-steps 900 \
    --env.time-limit-seconds 15 \
    --env.crash-height -10 \
    --env.safety-altitude -8 \
    --env.pos-bound 40 \
    --env.w-progress 35 \
    --env.w-gate 30 \
    --env.w-finish 180 \
    --env.w-time 0.15 \
    --env.w-cross-track 10 \
    --env.w-gate-crossing-error 30 \
    --env.gate-crossing-error-from-gate-index 0 \
    --env.invalid-penalty 600 \
    --train.total-timesteps "$TIMESTEPS" \
    --train.learning-rate "$LEARNING_RATE" \
    --train.min-lr-ratio 0.1 \
    --train.ent-coef 0 \
    --train.reward-scale 0.005 \
    --train.minibatch-size 1024 \
    --train.horizon 64 \
    --train.phase-prio-obs-index -1 \
    --train.phase-prio-scale 0 \
    --train.train-encoder-feature-start -1 \
    --train.train-encoder-feature-end -1 \
    "$@"
