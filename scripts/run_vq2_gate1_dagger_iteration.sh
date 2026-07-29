#!/usr/bin/env bash
# Generate a native two-gate teacher dataset and distill only its Gate-1 phase
# into a recurrent PufferLib policy. Privileged state is confined to labels;
# the checkpoint still receives the 32-value deployable observation ABI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-logs/drone_race_full_policy_six_gate_bootstrap/ppo_n104_phase_adapter/checkpoints/drone_race_full_policy_six_gate_bootstrap/1784295732298/0000000001048576.bin}"
STAGE_NAME="${STAGE_NAME:-vq2_n288_gate1_expert_bc}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/$STAGE_NAME}"
DATASET="${DATASET:-$RUN_ROOT/teacher.bin}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-$RUN_ROOT/policy.bin}"
EPISODES="${EPISODES:-256}"
STUDENT_PROBABILITY="${STUDENT_PROBABILITY:-0}"
ACTION_NOISE="${ACTION_NOISE:-0}"
EPOCHS="${EPOCHS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
ROBUST="${ROBUST:-0}"
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
    NUM_GATES=2
    GATE0_X=10.78 GATE0_Y=0.084 GATE0_Z=0.56
    GATE1_X=21.56 GATE1_Y=0.084 GATE1_Z=0.56
    GATE_RADIUS=0.75 GATE0_RADIUS=0.75 GATE1_RADIUS=0.75
    OBSERVABLE_GATE_PROGRESS=1
    OBSERVABLE_GATE_INDEX_DENOMINATOR=6
    OBSERVABLE_GATE_PHASE_ONEHOT=1
    SITL_GATE_OBS_SAMPLE_INTERVAL_STEPS=4
    TEACHER_GATE_DROPOUT_RANGE=4.25
    TEACHER_GATE_DROPOUT_FROM_GATE=0
    SITL_GATE_TRANSITION_MIN_FORWARD_SPEED=0.05
    TIME_LIMIT_SECONDS=20
    SEED=1288 EPISODE_SEED_SEQUENCE=1
)
if [[ "$ROBUST" == 1 ]]; then
    collector_env+=(
        START_X_JITTER=0.15 START_Y_JITTER=0.15 START_Z_JITTER=0.15
        START_VX_JITTER=0.10 START_VY_JITTER=0.10 START_VZ_JITTER=0.10
        START_ROLL_JITTER_RAD=0.02 START_PITCH_JITTER_RAD=0.02
        START_YAW_JITTER_RAD=0.02
        START_WX_JITTER=0.02 START_WY_JITTER=0.02 START_WZ_JITTER=0.02
        GATE_POSITION_DOMAIN_RANDOMIZE=1
        GATE_POSITION_DOMAIN_RANDOMIZE_PROBABILITY=1
        GATE_POSITION_RANDOMIZE_FROM_INDEX=0
        GATE_POSITION_JITTER_X=0.25
        GATE_POSITION_JITTER_Y=0.15
        GATE_POSITION_JITTER_Z=0.20
    )
elif [[ "$ROBUST" != 0 ]]; then
    echo "ROBUST must be 0 or 1" >&2
    exit 2
fi

# The collector requires at least two gates. Gate 2 is a straight extension
# used only to end episodes cleanly and to provide source-policy anchoring.
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
    --seed 1288 \
    --device cuda \
    --num-layers 3 \
    --checkpoint-layout-precision-bytes 4 \
    --parallel-sequence \
    --restore-best-training-loss \
    --gate-progress-observation \
    --min-race-phase 0 \
    --max-race-phase 0 \
    --source-anchor-weight 1 \
    --source-anchor-outside-active \
    --action-loss-weights 1 1 2 0.1
