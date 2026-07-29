#!/usr/bin/env bash
# Source-locked Vast.ai bootstrap and resumable VG002 oracle admission.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT to the pushed source commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_SKIP_APT="${VQ2_SKIP_APT:-0}"
VQ2_OUTPUT_ROOT="${VQ2_OUTPUT_ROOT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg002_variable_gate_oracle_admission}"

run_vg002_admission() {
    python scripts/eval_vq2_variable_gate_oracle.py \
        --output-root "$VQ2_OUTPUT_ROOT" \
        --counts 5 8 11 12 \
        --agents 512 \
        --episodes 512 \
        --seed 429020 \
        --resume
}

cd "$VQ2_WORKSPACE_PATH"

if [ "$(git rev-parse HEAD)" != "$VQ2_EXPECTED_COMMIT" ]; then
    echo "remote HEAD does not match VQ2_EXPECTED_COMMIT" >&2
    exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "remote tracked worktree is dirty" >&2
    exit 2
fi

# Once the evaluator has published its source/runtime state, resume without
# apt, pip, tests, or a rebuild. Rebuilding the extension could change its
# immutable binary hash and must fail closed instead of mutating a live run.
if [ -f "$VQ2_OUTPUT_ROOT/state.json" ]; then
    if [ ! -x .venv/bin/python ]; then
        echo "VG002 resume state exists but the source-locked venv is missing" >&2
        exit 2
    fi
    source .venv/bin/activate
    run_vg002_admission
    exit $?
fi

if [ "$VQ2_SKIP_APT" != "1" ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        clang \
        curl \
        git \
        libomp-dev \
        python3-dev \
        python3-pip \
        python3-venv \
        rsync \
        unzip
fi

for command_name in git clang nvcc nvidia-smi python3; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "missing required command: $command_name" >&2
        exit 2
    fi
done

VQ2_CPU_COUNT="$(nproc)"
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_GIB="$(df -BG --output=size "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_CPU_COUNT" -lt 32 ]; then
    echo "VG002 requires at least 32 visible CPUs; found $VQ2_CPU_COUNT" >&2
    exit 2
fi
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG002 requires approximately 64 GiB RAM; found ${VQ2_RAM_GIB} GiB" >&2
    exit 2
fi
if [ "$VQ2_DISK_GIB" -lt 95 ]; then
    echo "VG002 requires a 100 GiB-class disk; found ${VQ2_DISK_GIB} GiB" >&2
    exit 2
fi

VQ2_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$VQ2_GPU_NAME" in
    *RTX\ 4090*|*A100*|*RTX\ A6000*) ;;
    *) echo "unsupported VG002 GPU: $VQ2_GPU_NAME" >&2; exit 2 ;;
esac

if [ ! -x .venv/bin/python ]; then
    python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pytest
python -m pip install -e .

python - <<'PY'
import sys
import torch

if sys.version_info < (3, 10):
    raise SystemExit("VG002 requires Python 3.10+")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the rented GPU")
major, minor = (int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
if (major, minor) < (2, 9):
    raise SystemExit(f"VG002 requires torch>=2.9; found {torch.__version__}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
)"
echo "commit=$VQ2_EXPECTED_COMMIT cpus=$VQ2_CPU_COUNT ram_gib=$VQ2_RAM_GIB disk_gib=$VQ2_DISK_GIB"
echo "gpu=$VQ2_GPU_NAME cuda_home=$VQ2_CUDA_HOME nvcc_arch=$VQ2_NVCC_ARCH"
nvidia-smi
nvcc --version

scripts/test_drone_race_native_regressions.sh
scripts/test_drone_race_vision_native_regressions.sh
CUDA_HOME="$VQ2_CUDA_HOME" NVCC_ARCH="$VQ2_NVCC_ARCH" \
    bash build.sh drone_race_vision --float

python - <<'PY'
from pufferlib import _C

assert _C.env_name == "drone_race_vision", _C.env_name
assert _C.precision_bytes == 4, _C.precision_bytes
print(f"native_env={_C.env_name} precision_bytes={_C.precision_bytes}")
PY

python -m pytest -q \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase.py \
    tests/test_collect_vq2_oracle_bc_dataset.py \
    tests/test_eval_vq2_native_oracle.py \
    tests/test_eval_vq2_variable_gate_oracle.py \
    tests/test_verify_vq2_variable_gate_binding.py

run_vg002_admission
