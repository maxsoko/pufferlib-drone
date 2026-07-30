#!/usr/bin/env bash
# Source-locked Vast.ai launcher and epoch-resumable VG010 balanced refit.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT to the pushed source commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg010_variable_gate_source_balanced_refit_001}"
VQ2_STATE="$VQ2_OUTPUT/training_state.pt"
VQ2_CLEAN="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg003_variable_gate_legal_bc_dataset_256"
VQ2_DAGGER="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg009_variable_gate_dagger_round1_corrected_512"
VQ2_PARENT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg005_variable_gate_recurrent_bc_001"

run_vg010_refit() {
    python scripts/train_vq2_variable_gate_dagger_refit.py \
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
    echo "VG010 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

# Once epoch-zero state exists, do not rebuild, reinstall, or retest. Only its
# exact state/identity-checked training continuation may run.
if [ -f "$VQ2_STATE" ]; then
    run_vg010_refit
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
    echo "VG010 requires at least 32 visible CPUs; found $VQ2_CPU_COUNT" >&2
    exit 2
fi
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG010 requires approximately 64 GiB RAM; found ${VQ2_RAM_GIB} GiB" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 20 ]; then
    echo "VG010 requires at least 20 GiB free; found ${VQ2_DISK_FREE_GIB} GiB" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_CLEAN/report.json" | cut -d" " -f1)" = \
    b8069d05cec63038659893a4020fa61ca1276ae035bc1a2ea5754f75e2298e85
test "$(sha256sum "$VQ2_CLEAN/metadata.json" | cut -d" " -f1)" = \
    7a5d6359641c3e01f838e6fcdb57d32ddc18cc71ddbf1f8b3d0025309cabfe64
test "$(sha256sum "$VQ2_DAGGER/report.json" | cut -d" " -f1)" = \
    72c1e41d128f16ff082362e0dac2fb0cea36eef1a7d65e8e96f8ec31c3f50637
test "$(sha256sum "$VQ2_DAGGER/metadata.json" | cut -d" " -f1)" = \
    da20bd4ef93b7371712880787226a7f41b23785841d7307d501e0b7a8f829468
test "$(sha256sum "$VQ2_PARENT/policy_best.pt" | cut -d" " -f1)" = \
    f686a35dc923137bc40fd6feca459e52e5585d5b6fa047b7f77a171e2832a6c3
test "$(sha256sum "$VQ2_PARENT/report.json" | cut -d" " -f1)" = \
    bdcd2b38ea3537f2c250281c63f2e6aeca0ad87c97521339f098eec7ff474e5b
test "$(sha256sum docs/vq2_vg009_variable_gate_dagger_round1_admission_2026-07-30.json | cut -d" " -f1)" = \
    71c4898f7a7a306b0f6def06bda63d5192de456c8a849e1f08f83ba908cd3091
test "$(sha256sum docs/vq2_variable_gate_solve_first_goal_prompt_2026-07-28.md | cut -d" " -f1)" = \
    052ba7cb7a6db0b0274731f555a726994f83d424cb50e70f055e804926333e89

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG010 cannot see CUDA")
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
)"

python - <<'PY'
import torch

name = torch.cuda.get_device_name(0)
if not any(candidate in name for candidate in ("RTX 4090", "A100", "RTX A6000")):
    raise SystemExit(f"unsupported VG010 GPU: {name}")
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

# Canonicalize only the CPU-side regression reductions on wide AMD hosts.
# CUDA/default remains in force for the actual refit below.
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python -m pytest -q \
    tests/test_vq2_recurrent_phase.py \
    tests/test_train_vq2_recurrent_bc.py \
    tests/test_train_vq2_variable_gate_recurrent_bc.py \
    tests/test_train_vq2_variable_gate_dagger_refit.py

python scripts/train_vq2_variable_gate_dagger_refit.py \
    --output "$VQ2_OUTPUT" \
    --device cuda
