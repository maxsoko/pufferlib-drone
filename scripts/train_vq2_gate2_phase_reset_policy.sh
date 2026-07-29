#!/usr/bin/env bash
# Native PPO for a zero-state PufferLib Gate-2 policy. The optional analytic
# action target and blend exist only inside training; deterministic evaluation
# and deployment execute the recurrent policy mean with no teacher assistance.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/vq2_n294_full_blend_refine/alpha_0p60.bin}"
STAGE_NAME="${STAGE_NAME:-vq2_n305_gate2_phase_reset_ppo_assist100}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/$STAGE_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/train_logs}"
TIMESTEPS="${TIMESTEPS:-1048576}"
GATE_RADIUS="${GATE_RADIUS:-0.75}"
TEACHER_BLEND="${TEACHER_BLEND:-1.0}"
TEACHER_REWARD="${TEACHER_REWARD:-20.0}"

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
    --seed 304 \
    --vec.total-agents 128 --vec.num-buffers 4 --vec.num-threads 8 \
    --env.num-gates 2 \
    --env.use-custom-start 1 \
    --env.start-gate-index 1 \
    --env.start-elapsed-time 3.25 \
    --env.mixed-start-curriculum 0 \
    --env.start-x 0 --env.start-y 0 --env.start-z 0 \
    --env.start-vx 4.677 --env.start-vy 0.006 --env.start-vz 0.173 \
    --env.start-qw 0.998520 --env.start-qx 0.052695 \
    --env.start-qy -0.013450 --env.start-qz 0.000931 \
    --env.start-wx 0.1226 --env.start-wy -0.0006 --env.start-wz 0 \
    --env.start-x-jitter 0 --env.start-y-jitter 0 --env.start-z-jitter 0 \
    --env.start-vx-jitter 0 --env.start-vy-jitter 0 --env.start-vz-jitter 0 \
    --env.start-roll-jitter-rad 0 \
    --env.start-pitch-jitter-rad 0 \
    --env.start-yaw-jitter-rad 0 \
    --env.start-wx-jitter 0 --env.start-wy-jitter 0 --env.start-wz-jitter 0 \
    --env.use-custom-gate-layout 1 \
    --env.gate0-x 0 --env.gate0-y 0 --env.gate0-z 0 \
    --env.gate1-x 14.74 --env.gate1-y 8.70 --env.gate1-z 1.095 \
    --env.gate-radius "$GATE_RADIUS" \
    --env.gate0-radius "$GATE_RADIUS" \
    --env.gate1-radius "$GATE_RADIUS" \
    --env.gate-position-domain-randomize 0 \
    --env.observable-gate-progress 1 \
    --env.observable-gate-index-denominator 6 \
    --env.observable-gate-phase-onehot 1 \
    --env.sitl-plant-domain-randomize 0 \
    --env.sitl-gate-obs-sample-interval-steps 4 \
    --env.sitl-gate-obs-dropout-range-m 4.25 \
    --env.sitl-gate-obs-dropout-from-index 1 \
    --env.max-steps 1200 \
    --env.time-limit-seconds 20 \
    --env.crash-height -10 \
    --env.safety-altitude -8 \
    --env.pos-bound 45 \
    --env.w-progress 35 \
    --env.w-gate 30 \
    --env.w-finish 180 \
    --env.w-time 0.15 \
    --env.w-cross-track 20 \
    --env.w-gate-crossing-error 30 \
    --env.gate-crossing-error-from-gate-index 1 \
    --env.w-action-teacher "$TEACHER_REWARD" \
    --env.teacher-action-blend "$TEACHER_BLEND" \
    --env.teacher-pitch-from-gate-index 1 \
    --env.teacher-pitch-speed-control 1 \
    --env.teacher-pitch-speed-target-m-s 1.30 \
    --env.teacher-pitch-speed-gain 0.35 \
    --env.teacher-pitch-speed-scale 0.24 \
    --env.teacher-roll-from-gate-index 1 \
    --env.teacher-roll-until-gate-index 2 \
    --env.teacher-thrust-from-gate-index 1 \
    --env.teacher-yaw-control 1 \
    --env.teacher-yaw-from-gate-index 1 \
    --env.teacher-yaw-action 0 \
    --env.teacher-roll-per-m 0.35 \
    --env.teacher-roll-rate-per-m-s 0.10 \
    --env.teacher-thrust-bias 0.05 \
    --env.teacher-thrust-per-m 0.30 \
    --env.teacher-thrust-rate-per-m-s 0.10 \
    --env.teacher-thrust-world-frame 1 \
    --env.teacher-pitch-weight 1 \
    --env.teacher-roll-weight 1 \
    --env.teacher-thrust-weight 1 \
    --env.teacher-yaw-weight 1 \
    --env.invalid-penalty 600 \
    --train.total-timesteps "$TIMESTEPS" \
    --train.learning-rate 1e-4 \
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
