#!/usr/bin/env bash
# Source-locked Vast.ai bootstrap and resumable VG003 legal corpus collection.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT to the pushed source commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg003_variable_gate_legal_bc_dataset_256}"
VQ2_STATE="${VQ2_OUTPUT}_state.json"

run_vg003_collection() {
    python scripts/collect_vq2_variable_gate_oracle_bc_dataset.py \
        --output "$VQ2_OUTPUT" \
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
if [ ! -x .venv/bin/python ]; then
    echo "VG003 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

# Once collection state exists, do not rebuild, reinstall, or retest. An
# interrupted attempt may only restart its exact state-locked collection.
if [ -f "$VQ2_STATE" ]; then
    run_vg003_collection
    exit $?
fi

for command_name in git clang nvcc nvidia-smi python; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "missing required command: $command_name" >&2
        exit 2
    fi
done

VQ2_CPU_COUNT="$(nproc)"
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_CPU_COUNT" -lt 32 ]; then
    echo "VG003 requires at least 32 visible CPUs; found $VQ2_CPU_COUNT" >&2
    exit 2
fi
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG003 requires approximately 64 GiB RAM; found ${VQ2_RAM_GIB} GiB" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 55 ]; then
    echo "VG003 requires at least 55 GiB free; found ${VQ2_DISK_FREE_GIB} GiB" >&2
    exit 2
fi

VQ2_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
case "$VQ2_GPU_NAME" in
    *RTX\ 4090*|*A100*|*RTX\ A6000*) ;;
    *) echo "unsupported VG003 GPU: $VQ2_GPU_NAME" >&2; exit 2 ;;
esac

python - <<'PY'
import sys
import torch

if sys.version_info < (3, 10):
    raise SystemExit("VG003 requires Python 3.10+")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the rented GPU")
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
)"
echo "commit=$VQ2_EXPECTED_COMMIT cpus=$VQ2_CPU_COUNT ram_gib=$VQ2_RAM_GIB disk_free_gib=$VQ2_DISK_FREE_GIB"
echo "gpu=$VQ2_GPU_NAME cuda_home=$VQ2_CUDA_HOME nvcc_arch=$VQ2_NVCC_ARCH"

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
    tests/test_collect_vq2_variable_gate_oracle_bc_dataset.py \
    tests/test_eval_vq2_native_oracle.py \
    tests/test_eval_vq2_variable_gate_oracle.py \
    tests/test_verify_vq2_variable_gate_binding.py

run_vg003_collection
