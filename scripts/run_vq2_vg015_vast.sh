#!/usr/bin/env bash
# Source-locked Vast.ai bootstrap and resumable VG015 teacher-free screen.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT to the pushed source commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg015_variable_gate_recurrent_teacher_free_256}"
VQ2_STATE="$VQ2_OUTPUT/state.json"
VQ2_CANDIDATE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg014_variable_gate_three_source_refit_001"
VQ2_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg014_three_source_refit_admission_2026-07-30.json"

run_vg015_screen() {
    python scripts/eval_vq2_vg015_variable_gate_recurrent_policy.py \
        --output "$VQ2_OUTPUT" \
        --device cuda \
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
    echo "VG015 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

# Once screen state exists, do not rebuild, reinstall, or retest. Resume only
# the exact missing count under its persisted source identity.
if [ -f "$VQ2_STATE" ]; then
    run_vg015_screen
    exit $?
fi

for command_name in git clang ccache nvcc nvidia-smi python; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "missing required command: $command_name" >&2
        exit 2
    fi
done
printf '#include <omp.h>\n' | clang -fopenmp -x c - -fsyntax-only

VQ2_CPU_COUNT="$(nproc)"
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_CPU_COUNT" -lt 32 ]; then
    echo "VG015 requires at least 32 visible CPUs; found $VQ2_CPU_COUNT" >&2
    exit 2
fi
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG015 requires approximately 64 GiB RAM; found ${VQ2_RAM_GIB} GiB" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 15 ]; then
    echo "VG015 requires at least 15 GiB free; found ${VQ2_DISK_FREE_GIB} GiB" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_CANDIDATE/policy_best.pt" | cut -d" " -f1)" = \
    60d69a1f80a1551b190cd4b517dfa029bc3dbe48728a853a47b658df0866d79e
test "$(sha256sum "$VQ2_CANDIDATE/report.json" | cut -d" " -f1)" = \
    067d9c198c14959dbb7221d34f9fec0f065a3cee41043e3ce0ff1f9a7bbf0a16
test "$(sha256sum "$VQ2_ADMISSION" | cut -d" " -f1)" = \
    073fc31e143e2d8e61e40cead685698c47e012b7153b32622bd4dc202fbcfa72

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG015 cannot see CUDA")
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
)"

python - <<'PY'
import torch

name = torch.cuda.get_device_name(0)
if not any(candidate in name for candidate in ("RTX 4090", "A100", "RTX A6000")):
    raise SystemExit(f"unsupported VG015 GPU: {name}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={name}")
PY

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

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python -m pytest -q \
    tests/test_eval_vq2_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg015_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_variable_gate_oracle.py \
    tests/test_eval_vq2_recurrent_policy.py \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase.py

python scripts/eval_vq2_vg015_variable_gate_recurrent_policy.py \
    --output "$VQ2_OUTPUT" \
    --device cuda
