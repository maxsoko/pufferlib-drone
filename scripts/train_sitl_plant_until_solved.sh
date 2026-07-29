#!/usr/bin/env bash
# Train drone_race_sitl_plant on WSL ext4 until native promotion metrics are met,
# then print the latest checkpoint for official-sim transfer.
set -euo pipefail

ROOT="${REPO_ROOT:-$HOME/pufferlib-drone}"
cd "$ROOT"
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_ARCH=sm_86

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    bash build.sh drone_race
fi

ENV_NAME="${ENV_NAME:-drone_race_sitl_plant}"
CHUNK_STEPS="${CHUNK_STEPS:-2000000}"
MAX_CHUNKS="${MAX_CHUNKS:-50}"
SUCCESS_TARGET="${SUCCESS_TARGET:-0.85}"
CRASH_TARGET="${CRASH_TARGET:-0.10}"
EVAL_EPISODES="${EVAL_EPISODES:-4096}"
LOAD_PATH="${LOAD_PATH:-}"
FAIL_FAST_CHUNKS="${FAIL_FAST_CHUNKS:-2}"
FAIL_FAST_SUCCESS_MAX="${FAIL_FAST_SUCCESS_MAX:-0.001}"
FAIL_FAST_CRASH_MIN="${FAIL_FAST_CRASH_MIN:-0.99}"
# Set FRESH_START=1 to skip auto-resume from latest checkpoint (avoids bad resume heap crashes).
FRESH_START="${FRESH_START:-0}"

latest_ckpt() {
    python - <<'PY'
import glob
import os
import sys

env_name = os.environ.get("ENV_NAME", "drone_race_sitl_plant")
pattern = os.path.join("checkpoints", env_name, "**", "*.bin")
candidates = glob.glob(pattern, recursive=True)
if not candidates:
    sys.exit(0)
best = max(
    candidates,
    key=lambda p: (
        int(os.path.basename(os.path.dirname(p))),
        int(os.path.basename(p).split(".")[0]),
    ),
)
print(best)
PY
}

chunk=0
catastrophic_chunks=0
echo "train-until-solved: MAX_CHUNKS=${MAX_CHUNKS} CHUNK_STEPS=${CHUNK_STEPS}" >&2
while [ "$chunk" -lt "$MAX_CHUNKS" ]; do
    chunk=$((chunk + 1))
    load_args=()
    ckpt="$(latest_ckpt)"
    resume=""
    if [ -n "$LOAD_PATH" ]; then
        resume="$LOAD_PATH"
        LOAD_PATH=""
    elif [ "$FRESH_START" != "1" ] && [ -n "$ckpt" ]; then
        resume="$ckpt"
    fi
    TIMESTEPS="$CHUNK_STEPS" ENV_NAME="$ENV_NAME" LOAD_PATH="$resume" SKIP_PIP=1 SKIP_BUILD=1 \
        bash scripts/train_drone_race_sitl_plant_local.sh || {
        echo "training chunk failed; retrying fresh (no resume — avoids heap crash on load)" >&2
        FRESH_START=1
        LOAD_PATH=""
        sleep 2
        continue
    }

    ckpt="$(latest_ckpt)"
    if [ -z "$ckpt" ]; then
        echo "no checkpoint after chunk ${chunk}" >&2
        continue
    fi

    echo "==> eval ${ckpt}"
    set +e
    eval_json="logs/${ENV_NAME}_goal_eval_chunk_${chunk}.json"
    python scripts/eval_drone_race_checkpoint.py "$ckpt" \
        --env-name "$ENV_NAME" \
        --eval-episodes "$EVAL_EPISODES" \
        --json-path "$eval_json"
    eval_status=$?
    set -e
    if [ "$eval_status" -ne 0 ]; then
        echo "eval failed; continuing training" >&2
        LOAD_PATH="$ckpt"
        continue
    fi

    success="$(python - <<PY
import json
d=json.load(open("$eval_json"))
m=d.get("metrics", d)
print(m.get("env/success_rate", m.get("success_rate", 0)))
PY
)"
    crash="$(python - <<PY
import json
d=json.load(open("$eval_json"))
m=d.get("metrics", d)
print(m.get("env/crash", m.get("crash", 1)))
PY
)"

    echo "chunk ${chunk} metrics: success_rate=${success} crash=${crash}"
    set +e
    python - <<PY
success=float("$success")
crash=float("$crash")
import sys
sys.exit(0 if success <= float("$FAIL_FAST_SUCCESS_MAX") and crash >= float("$FAIL_FAST_CRASH_MIN") else 1)
PY
    catastrophic=$?
    set -e
    if [ "$catastrophic" -eq 0 ]; then
        catastrophic_chunks=$((catastrophic_chunks + 1))
    else
        catastrophic_chunks=0
    fi
    if [ "$FAIL_FAST_CHUNKS" -gt 0 ] && [ "$catastrophic_chunks" -ge "$FAIL_FAST_CHUNKS" ]; then
        echo "fail-fast: ${catastrophic_chunks} consecutive chunks at success<=${FAIL_FAST_SUCCESS_MAX}, crash>=${FAIL_FAST_CRASH_MIN}; change curriculum before retry" >&2
        latest_ckpt
        exit 2
    fi
    set +e
    python - <<PY
success=float("$success")
crash=float("$crash")
import sys
if success >= float("$SUCCESS_TARGET") and crash <= float("$CRASH_TARGET"):
    print("PROMOTION_THRESHOLDS_MET")
    sys.exit(0)
sys.exit(1)
PY
    promoted=$?
    set -e
    if [ "$promoted" -eq 0 ]; then
        echo "SOLVED native checkpoint: $ckpt"
        echo "$ckpt" > "logs/${ENV_NAME}_solved_checkpoint.txt"
        exit 0
    fi
    LOAD_PATH="$ckpt"
done

echo "max chunks (${MAX_CHUNKS}) reached without hitting promotion thresholds" >&2
latest_ckpt
exit 1
