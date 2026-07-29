#!/usr/bin/env bash
# N209: train one recurrent six-gate policy on ordinary full-course starts and
# a bounded distribution reconstructed from five official Gate-4 entries.
# Native-only: this script never launches, resets, or connects to FlightSim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n096_raw_anchor_mixture/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784286271692/0000000000491520.bin}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/n209_measured_entry_mixture}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/train_logs}"
TIMESTEPS="${TIMESTEPS:-1048576}"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    echo "missing raw six-gate parent: $SOURCE_CHECKPOINT" >&2
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
    --tag n209_measured_entry_mixture \
    --seed 209 \
    --env.use-custom-start 1 \
    --env.start-gate-index 3 \
    --env.start-elapsed-time 10.85 \
    --env.start-elapsed-time-jitter 0.55 \
    --env.mixed-start-curriculum 1 \
    --env.segment-start-probability 0.50 \
    --env.start-x 64.3 --env.start-y -0.5 --env.start-z -6.9 \
    --env.start-x-jitter 6.0 --env.start-y-jitter 5.0 --env.start-z-jitter 2.0 \
    --env.start-vx 8.5 --env.start-vy 0.4 --env.start-vz -5.0 \
    --env.start-vx-jitter 4.0 --env.start-vy-jitter 2.0 --env.start-vz-jitter 2.5 \
    --env.start-qw 0.99529518 --env.start-qx -0.03851571 \
    --env.start-qy 0.08873983 --env.start-qz 0.00541173 \
    --env.start-roll-jitter-rad 0.22 \
    --env.start-pitch-jitter-rad 0.04 \
    --env.start-yaw-jitter-rad 0.04 \
    --env.start-wx -0.40 --env.start-wy 0.33 --env.start-wz -0.05 \
    --env.start-wx-jitter 0.25 --env.start-wy-jitter 0.30 --env.start-wz-jitter 0.10 \
    --env.gate4-x 122 --env.gate4-y 5.10711 --env.gate4-z -18.24322 \
    --env.gate5-x 148 --env.gate5-y 5.10711 --env.gate5-z -18.24322 \
    --env.gate-radius 0.95 \
    --env.gate0-radius 0.75 --env.gate1-radius 0 --env.gate2-radius 0 \
    --env.gate3-radius 0.875 --env.gate4-radius 0 --env.gate5-radius 0.75 \
    --env.gate-position-domain-randomize 1 \
    --env.gate-position-domain-randomize-probability 0.5 \
    --env.gate-position-randomize-from-index 4 \
    --env.gate-position-jitter-x 2 \
    --env.gate-position-jitter-y 6 \
    --env.gate-position-jitter-z 3 \
    --env.observable-gate-progress 1 \
    --env.observable-gate-index-denominator 6 \
    --env.observable-gate-phase-onehot 0 \
    --env.sitl-plant-domain-randomize 0 \
    --env.sitl-gate-transition-min-forward-speed 4.1875 \
    --env.w-time 0 \
    --env.w-cross-track 10 \
    --env.cross-track-from-gate-index 3 \
    --env.w-gate-crossing-error 20 \
    --env.gate-crossing-error-from-gate-index 3 \
    --train.total-timesteps "$TIMESTEPS" \
    --train.learning-rate 1e-5 \
    --train.min-lr-ratio 0.1 \
    --train.ent-coef 0.0005 \
    --train.reward-scale 0.005 \
    --train.phase-prio-obs-index -1 \
    --train.phase-prio-scale 0 \
    "$@"
