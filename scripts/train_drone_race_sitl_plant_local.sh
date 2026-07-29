#!/usr/bin/env bash
# Train drone_race_sitl_plant on a local Linux GPU (WSL2 + RTX 3070).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-drone_race_sitl_plant}"
TIMESTEPS="${TIMESTEPS:-25000000}"
LOAD_PATH="${LOAD_PATH:-}"
FORCE_SLOWLY="${FORCE_SLOWLY:-0}"
SKIP_PIP="${SKIP_PIP:-0}"

if [ "$(uname -s)" != "Linux" ]; then
    echo "Run this script inside WSL2/Linux. Windows path: wsl bash scripts/train_drone_race_sitl_plant_local.sh" >&2
    exit 1
fi

case "$ROOT" in
    /mnt/*)
        echo "WARNING: training from NTFS mount ($ROOT) caused heap corruption in prior runs." >&2
        echo "         Sync to ext4 first: bash scripts/sync_wsl_ext4_repo.sh ~/pufferlib-drone" >&2
        if [ "${ALLOW_MNT_C_TRAINING:-0}" != "1" ]; then
            echo "Set ALLOW_MNT_C_TRAINING=1 to override, or run from ~/pufferlib-drone." >&2
            exit 1
        fi
        ;;
esac

pick_cuda_home() {
    for candidate in /usr/local/cuda-12.6 /usr/local/cuda-12 /usr/local/cuda; do
        if [ -x "$candidate/bin/nvcc" ]; then
            echo "$candidate"
            return 0
        fi
    done
    if command -v nvcc >/dev/null 2>&1; then
        dirname "$(dirname "$(command -v nvcc)")"
        return 0
    fi
    return 1
}

CUDA_HOME_SELECTED=""
if CUDA_HOME_SELECTED="$(pick_cuda_home)"; then
    export CUDA_HOME="$CUDA_HOME_SELECTED"
    export PATH="$CUDA_HOME/bin:$PATH"
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found; will use --slowly after CPU env build." >&2
fi

# RTX 3070 (Ampere sm_86). Older apt nvcc (11.x) rejects -arch=native.
export NVCC_ARCH="${NVCC_ARCH:-sm_86}"

if command -v ccache >/dev/null 2>&1; then
    export NVCC="ccache ${CUDA_HOME:-/usr}/bin/nvcc"
else
    export NVCC="${CUDA_HOME:-/usr}/bin/nvcc"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; GPU driver/WSL CUDA integration missing." >&2
    exit 1
fi

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [ "$SKIP_PIP" != "1" ]; then
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install nvidia-cudnn-cu12 nvidia-nccl-cu12 nvidia-cuda-nvcc-cu12 2>/dev/null || true
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | head -1
fi

bash scripts/test_drone_race_native_regressions.sh

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    train_extra=()
    if [ "$FORCE_SLOWLY" = "1" ]; then
        echo "FORCE_SLOWLY=1: building CPU env and using PyTorch backend"
        bash build.sh drone_race --cpu
        train_extra=(--slowly)
    else
        set +e
        bash build.sh drone_race
        build_status=$?
        set -e
        if [ "$build_status" -ne 0 ]; then
            echo "Native CUDA build failed (status=$build_status); falling back to --slowly" >&2
            bash build.sh drone_race --cpu
            train_extra=(--slowly)
        fi
    fi
else
    echo "SKIP_BUILD=1: using existing pufferlib/_C build"
    train_extra=()
    if [ "$FORCE_SLOWLY" = "1" ]; then
        train_extra=(--slowly)
    fi
fi

load_args=()
if [ -n "$LOAD_PATH" ]; then
    load_args=(--load-model-path "$LOAD_PATH")
fi

echo "==> training ${ENV_NAME} for ${TIMESTEPS} timesteps (interface_mode=2)"
python -m pufferlib.pufferl train "$ENV_NAME" \
    "${train_extra[@]}" \
    "${load_args[@]}" \
    --train.total-timesteps "$TIMESTEPS"

echo "latest checkpoint:"
python - <<'PY'
import glob, os
env = os.environ.get("ENV_NAME", "drone_race_sitl_plant")
cands = glob.glob(os.path.join("checkpoints", env, "**", "*.bin"), recursive=True)
if cands:
    best = max(
        cands,
        key=lambda p: (
            int(os.path.basename(os.path.dirname(p))),
            int(os.path.basename(p).split(".")[0]),
        ),
    )
    print(best)
PY
