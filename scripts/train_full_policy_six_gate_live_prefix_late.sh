#!/usr/bin/env bash
# N112: preserve the live-proven Gates-1-through-3 recurrent policy exactly and
# train only one-hot encoder columns for Gates 4-6 from native segment starts.
# Native-only: this script never connects to or resets official FlightSim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/mnt/c/Users/anon/code/pufferlib-drone/checkpoints/drone_race_full_policy_gate3_visual/v6c_latest_obsmatch_r135_dropout_e10.bin}"
RUN_ROOT="${RUN_ROOT:-logs/drone_race_full_policy_six_gate_bootstrap/n112_live_gate3_parent}"
EXPANDED_CHECKPOINT="${EXPANDED_CHECKPOINT:-$RUN_ROOT/source/gate3_live_expanded32_fp32.bin}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/train_logs}"
TIMESTEPS="${TIMESTEPS:-1048576}"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    echo "missing live-prefix source checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 2
fi
if pgrep -af 'pufferlib.pufferl train' | grep -v "$$" >/dev/null; then
    echo "another PufferLib training process is already running" >&2
    exit 2
fi

if [[ ! -f "$EXPANDED_CHECKPOINT" ]]; then
    python scripts/resize_policy_checkpoint_input.py \
        "$SOURCE_CHECKPOINT" "$EXPANDED_CHECKPOINT" \
        --source-input-dim 23 --target-input-dim 32 \
        --source-precision-bytes 2 --target-precision-bytes 4
fi

python -m pufferlib.pufferl train drone_race_full_policy_six_gate_bootstrap \
    --load-model-path "$EXPANDED_CHECKPOINT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --checkpoint-interval 1 \
    --eval-episodes 0 \
    --tag n112_live_gate3_prefix_late_columns \
    --seed 112 \
    --env.observable-gate-progress 1 \
    --env.observable-gate-index-denominator 6 \
    --env.observable-gate-phase-onehot 1 \
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
