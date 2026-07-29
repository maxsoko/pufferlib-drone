#!/usr/bin/env bash
# Capture a native-only Gate-2 trace for iterative PufferLib residual
# distillation. The teacher blend is training infrastructure; the resulting
# student is always screened again with teacher_action_blend=0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to an FP32 PufferLib checkpoint}"
OUTPUT="${OUTPUT:?set OUTPUT to the output NPZ path}"
TEACHER_BLEND="${TEACHER_BLEND:-0.20}"
GATE_RADIUS="${GATE_RADIUS:-2.0}"
TOTAL_AGENTS="${TOTAL_AGENTS:-128}"
HORIZON="${HORIZON:-64}"
MAX_ROLLOUTS="${MAX_ROLLOUTS:-20}"

python scripts/capture_native_policy_trace.py "$CHECKPOINT" "$OUTPUT" \
    --floor 0.05 \
    --total-agents "$TOTAL_AGENTS" \
    --horizon "$HORIZON" \
    --max-rollouts "$MAX_ROLLOUTS" \
    --env-name drone_race_full_policy_six_gate_bootstrap \
    --layout-precision-bytes 4 \
    --config-override=--env.num-gates=2 \
    --config-override=--env.use-custom-start=1 \
    --config-override=--env.start-gate-index=1 \
    --config-override=--env.start-elapsed-time=3.25 \
    --config-override=--env.mixed-start-curriculum=0 \
    --config-override=--env.start-x=0 \
    --config-override=--env.start-y=0 \
    --config-override=--env.start-z=0 \
    --config-override=--env.start-vx=4.677 \
    --config-override=--env.start-vy=0.006 \
    --config-override=--env.start-vz=0.173 \
    --config-override=--env.start-qw=0.998520 \
    --config-override=--env.start-qx=0.052695 \
    --config-override=--env.start-qy=-0.013450 \
    --config-override=--env.start-qz=0.000931 \
    --config-override=--env.start-wx=0.1226 \
    --config-override=--env.start-wy=-0.0006 \
    --config-override=--env.start-wz=0 \
    --config-override=--env.start-elapsed-time-jitter=0 \
    --config-override=--env.start-x-jitter=0 \
    --config-override=--env.start-y-jitter=0 \
    --config-override=--env.start-z-jitter=0 \
    --config-override=--env.start-vx-jitter=0 \
    --config-override=--env.start-vy-jitter=0 \
    --config-override=--env.start-vz-jitter=0 \
    --config-override=--env.start-roll-jitter-rad=0 \
    --config-override=--env.start-pitch-jitter-rad=0 \
    --config-override=--env.start-yaw-jitter-rad=0 \
    --config-override=--env.start-wx-jitter=0 \
    --config-override=--env.start-wy-jitter=0 \
    --config-override=--env.start-wz-jitter=0 \
    --config-override=--env.use-custom-gate-layout=1 \
    --config-override=--env.gate0-x=0 \
    --config-override=--env.gate0-y=0 \
    --config-override=--env.gate0-z=0 \
    --config-override=--env.gate1-x=14.74 \
    --config-override=--env.gate1-y=8.70 \
    --config-override=--env.gate1-z=1.095 \
    --config-override=--env.gate-radius="$GATE_RADIUS" \
    --config-override=--env.gate0-radius="$GATE_RADIUS" \
    --config-override=--env.gate1-radius="$GATE_RADIUS" \
    --config-override=--env.observable-gate-progress=1 \
    --config-override=--env.observable-gate-index-denominator=6 \
    --config-override=--env.observable-gate-phase-onehot=1 \
    --config-override=--env.sitl-plant-domain-randomize=0 \
    --config-override=--env.sitl-gate-obs-sample-interval-steps=4 \
    --config-override=--env.sitl-gate-obs-dropout-range-m=4.25 \
    --config-override=--env.sitl-gate-obs-dropout-from-index=1 \
    --config-override=--env.max-steps=1200 \
    --config-override=--env.time-limit-seconds=20 \
    --config-override=--env.crash-height=-10 \
    --config-override=--env.safety-altitude=-8 \
    --config-override=--env.pos-bound=45 \
    --config-override=--env.teacher-action-blend="$TEACHER_BLEND" \
    --config-override=--env.teacher-pitch-from-gate-index=1 \
    --config-override=--env.teacher-pitch-speed-control=1 \
    --config-override=--env.teacher-pitch-speed-target-m-s=1.30 \
    --config-override=--env.teacher-pitch-speed-gain=0.35 \
    --config-override=--env.teacher-pitch-speed-scale=0.24 \
    --config-override=--env.teacher-roll-from-gate-index=1 \
    --config-override=--env.teacher-roll-until-gate-index=2 \
    --config-override=--env.teacher-thrust-from-gate-index=1 \
    --config-override=--env.teacher-yaw-control=1 \
    --config-override=--env.teacher-yaw-from-gate-index=1 \
    --config-override=--env.teacher-yaw-action=0 \
    --config-override=--env.teacher-roll-per-m=0.35 \
    --config-override=--env.teacher-roll-rate-per-m-s=0.10 \
    --config-override=--env.teacher-thrust-bias=0 \
    --config-override=--env.teacher-thrust-per-m=0.30 \
    --config-override=--env.teacher-thrust-rate-per-m-s=0.10 \
    --config-override=--env.teacher-thrust-world-frame=1
