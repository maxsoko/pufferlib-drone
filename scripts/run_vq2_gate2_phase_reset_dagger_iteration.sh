#!/usr/bin/env bash
# Train a phase-reset recurrent PufferLib policy from N295's measured official
# Gate-1 exit. Native state exists only in teacher labels and reset sampling;
# the student still consumes the legal 32-value deployment observation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/vq2_n294_full_blend_refine/alpha_0p60.bin}"
STAGE_NAME="${STAGE_NAME:-vq2_n297_gate2_phase_reset_expert_bc}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/$STAGE_NAME}"
DATASET="${DATASET:-$RUN_ROOT/teacher.bin}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-$RUN_ROOT/policy.bin}"
EPISODES="${EPISODES:-256}"
STUDENT_PROBABILITY="${STUDENT_PROBABILITY:-0}"
ACTION_NOISE="${ACTION_NOISE:-0}"
EPOCHS="${EPOCHS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
ROBUST="${ROBUST:-0}"
KEEP_SUCCESSFUL_ONLY="${KEEP_SUCCESSFUL_ONLY:-1}"
EXTRA_DATASET="${EXTRA_DATASET:-}"

if [[ ! -f "$PARENT_CHECKPOINT" ]]; then
    echo "missing parent checkpoint: $PARENT_CHECKPOINT" >&2
    exit 2
fi
if [[ "$KEEP_SUCCESSFUL_ONLY" != 0 && "$KEEP_SUCCESSFUL_ONLY" != 1 ]]; then
    echo "KEEP_SUCCESSFUL_ONLY must be 0 or 1" >&2
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
    START_GATE_INDEX=1
    START_ELAPSED_TIME=3.25
    START_X=0 START_Y=0 START_Z=0
    START_VX=4.677 START_VY=0.006 START_VZ=0.173
    START_QW=0.998520 START_QX=0.052695
    START_QY=-0.013450 START_QZ=0.000931
    START_WX=0.1226 START_WY=-0.0006 START_WZ=0
    GATE0_X=0 GATE0_Y=0 GATE0_Z=0
    # Legal N295 camera reconstruction at the Gate-1 index transition.
    GATE1_X=14.74 GATE1_Y=8.70 GATE1_Z=1.095
    GATE_RADIUS=0.75 GATE0_RADIUS=0.75 GATE1_RADIUS=0.75
    OBSERVABLE_GATE_PROGRESS=1
    OBSERVABLE_GATE_INDEX_DENOMINATOR=6
    OBSERVABLE_GATE_PHASE_ONEHOT=1
    SITL_GATE_OBS_SAMPLE_INTERVAL_STEPS=4
    TEACHER_GATE_DROPOUT_RANGE=4.25
    TEACHER_GATE_DROPOUT_FROM_GATE=1
    SITL_GATE_TRANSITION_MIN_FORWARD_SPEED=0.05
    TIME_LIMIT_SECONDS=20
    SEED=1298 EPISODE_SEED_SEQUENCE=1
    POLICY_TEACHER=1
    TEACHER_FEEDBACK_FROM_GATE=1
    TEACHER_FEEDBACK_UNTIL_GATE=2
    TEACHER_FEEDBACK_ROLL_KP=0.18
    TEACHER_FEEDBACK_ROLL_KD=0.10
    TEACHER_FEEDBACK_THRUST_KP=0.12
    TEACHER_FEEDBACK_THRUST_KD=0.10
    TEACHER_FEEDBACK_THRUST_BIAS=0.05
    TEACHER_ANALYTIC_PITCH_FROM_GATE=1
    TEACHER_KEEP_SUCCESSFUL_EPISODES_ONLY="$KEEP_SUCCESSFUL_ONLY"
)
if [[ "$ROBUST" == 1 ]]; then
    collector_env+=(
        START_ELAPSED_TIME_JITTER=0.15
        START_X_JITTER=0.10 START_Y_JITTER=0.10 START_Z_JITTER=0.10
        START_VX_JITTER=0.20 START_VY_JITTER=0.10 START_VZ_JITTER=0.10
        START_ROLL_JITTER_RAD=0.015
        START_PITCH_JITTER_RAD=0.015
        START_YAW_JITTER_RAD=0.015
        START_WX_JITTER=0.03 START_WY_JITTER=0.03 START_WZ_JITTER=0.03
        GATE_POSITION_DOMAIN_RANDOMIZE=1
        GATE_POSITION_DOMAIN_RANDOMIZE_PROBABILITY=1
        GATE_POSITION_RANDOMIZE_FROM_INDEX=1
        GATE_POSITION_JITTER_X=0.30
        GATE_POSITION_JITTER_Y=0.40
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
    --seed 1298 \
    --device cuda \
    --num-layers 3 \
    --checkpoint-layout-precision-bytes 4 \
    --parallel-sequence \
    --restore-best-training-loss \
    --gate-progress-observation \
    --min-race-phase 0.166 \
    --max-race-phase 0.167 \
    --action-loss-weights 1 1 2 0.1
