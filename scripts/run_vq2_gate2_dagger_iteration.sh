#!/usr/bin/env bash
# Distill the camera-measured VQ2 Gate-2 turn into one recurrent PufferLib
# policy. Privileged state is confined to native teacher labels; deployment
# still receives only the legal 32-value camera/IMU/action/progress ABI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/vq2_n294_full_blend_refine/alpha_0p60.bin}"
STAGE_NAME="${STAGE_NAME:-vq2_n296_gate2_expert_bc}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/$STAGE_NAME}"
DATASET="${DATASET:-$RUN_ROOT/teacher.bin}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-$RUN_ROOT/policy.bin}"
EPISODES="${EPISODES:-256}"
STUDENT_PROBABILITY="${STUDENT_PROBABILITY:-0}"
ACTION_NOISE="${ACTION_NOISE:-0}"
EPOCHS="${EPOCHS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
ROBUST="${ROBUST:-1}"
EXTRA_DATASET="${EXTRA_DATASET:-}"

if [[ ! -f "$PARENT_CHECKPOINT" ]]; then
    echo "missing parent checkpoint: $PARENT_CHECKPOINT" >&2
    exit 2
fi
mkdir -p build "$RUN_ROOT"
if [[ ! -x build/collect_full_policy_teacher \
        || scripts/collect_full_policy_teacher.c -nt build/collect_full_policy_teacher \
        || ocean/drone_race/drone_race.c -nt build/collect_full_policy_teacher ]]; then
    clang -O2 -DNDEBUG -I. scripts/collect_full_policy_teacher.c \
        -o build/collect_full_policy_teacher -lm
fi

collector_env=(
    CHECKPOINT_LAYOUT_PRECISION_BYTES=4
    POLICY_HZ=60
    NUM_GATES=2
    GATE0_X=10.78 GATE0_Y=0.084 GATE0_Z=0.56
    # Gate 2 is reconstructed only from N295's legal post-crossing camera pose.
    GATE1_X=25.13 GATE1_Y=8.96 GATE1_Z=1.65
    GATE_RADIUS=0.75 GATE0_RADIUS=0.75 GATE1_RADIUS=0.75
    OBSERVABLE_GATE_PROGRESS=1
    OBSERVABLE_GATE_INDEX_DENOMINATOR=6
    OBSERVABLE_GATE_PHASE_ONEHOT=1
    SITL_GATE_OBS_SAMPLE_INTERVAL_STEPS=4
    TEACHER_GATE_DROPOUT_RANGE=4.25
    TEACHER_GATE_DROPOUT_FROM_GATE=0
    SITL_GATE_TRANSITION_MIN_FORWARD_SPEED=0.05
    TIME_LIMIT_SECONDS=25
    SEED=1296 EPISODE_SEED_SEQUENCE=1
    POLICY_TEACHER=1
    TEACHER_FEEDBACK_FROM_GATE=1
    TEACHER_FEEDBACK_UNTIL_GATE=2
    TEACHER_FEEDBACK_ROLL_KP=0.12
    TEACHER_FEEDBACK_ROLL_KD=0.20
    TEACHER_FEEDBACK_THRUST_KP=0.04
    TEACHER_FEEDBACK_THRUST_KD=0.25
    TEACHER_ANALYTIC_PITCH_FROM_GATE=1
)
if [[ "$ROBUST" == 1 ]]; then
    collector_env+=(
        GATE_POSITION_DOMAIN_RANDOMIZE=1
        GATE_POSITION_DOMAIN_RANDOMIZE_PROBABILITY=1
        GATE_POSITION_RANDOMIZE_FROM_INDEX=1
        GATE_POSITION_JITTER_X=0.35
        GATE_POSITION_JITTER_Y=0.50
        GATE_POSITION_JITTER_Z=0.30
    )
elif [[ "$ROBUST" != 0 ]]; then
    echo "ROBUST must be 0 or 1" >&2
    exit 2
fi

env "${collector_env[@]}" build/collect_full_policy_teacher \
    "$DATASET" "$EPISODES" "$PARENT_CHECKPOINT" \
    "$STUDENT_PROBABILITY" "$ACTION_NOISE" 0

dataset_args=(--dataset "$DATASET")
if [[ -n "$EXTRA_DATASET" ]]; then
    if [[ ! -f "$EXTRA_DATASET" ]]; then
        echo "missing extra dataset: $EXTRA_DATASET" >&2
        exit 2
    fi
    dataset_args+=(--dataset "$EXTRA_DATASET")
fi

python scripts/train_full_policy_bc.py \
    "$PARENT_CHECKPOINT" "$OUTPUT_CHECKPOINT" \
    "${dataset_args[@]}" \
    --epochs "$EPOCHS" \
    --batch-episodes 16 \
    --chunk-length 128 \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay 0 \
    --max-grad-norm 1 \
    --seed 1296 \
    --device cuda \
    --num-layers 3 \
    --checkpoint-layout-precision-bytes 4 \
    --parallel-sequence \
    --restore-best-training-loss \
    --gate-progress-observation \
    --min-race-phase 0.166 \
    --max-race-phase 0.167 \
    --source-anchor-weight 1 \
    --source-anchor-outside-active \
    --action-loss-weights 1 1 2 0.1
