#!/usr/bin/env bash
# Source-locked Vast bootstrap and epoch-resumable VG028 seven-source refit.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG028 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg028_variable_gate_seven_source_refit_001}"
VQ2_STATE="$VQ2_OUTPUT/training_state.pt"
VQ2_VG027="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg027_variable_gate_dagger_round6_vg025_visited_512"
VQ2_VG027_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg027_variable_gate_dagger_round6_admission_2026-07-30.json"
VQ2_PARENT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg025_variable_gate_six_source_refit_001"

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
    echo "VG028 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

run_training() {
    if [ -f "$VQ2_STATE" ]; then
        python scripts/train_vq2_variable_gate_seven_source_refit.py \
            --output "$VQ2_OUTPUT" \
            --device cuda \
            --resume
    else
        python scripts/train_vq2_variable_gate_seven_source_refit.py \
            --output "$VQ2_OUTPUT" \
            --device cuda
    fi
}

# Epoch state already binds sources, runtime, RNG, optimizer, and every corpus.
if [ -f "$VQ2_STATE" ]; then
    run_training
    exit $?
fi

for command_name in git clang ccache nvcc nvidia-smi python; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "missing required command: $command_name" >&2
        exit 2
    fi
done
printf '#include <omp.h>\n' | clang -fopenmp -x c - -fsyntax-only
if [ "$(nproc)" -lt 32 ]; then
    echo "VG028 requires at least 32 visible CPUs" >&2
    exit 2
fi
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG028 requires approximately 64 GiB RAM" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 15 ]; then
    echo "VG028 requires at least 15 GiB free" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_PARENT/policy_best.pt" | cut -d" " -f1)" = \
    85671c31a7c7a25cf49414cd51259357a2efd64b5d2b441fea27c51b93bf77aa
test "$(sha256sum "$VQ2_PARENT/report.json" | cut -d" " -f1)" = \
    b9d22d2867b8518a0f8b1b8a0fbbad9d0977f3a57e189b83d1362be98f38dcb6
test "$(sha256sum "$VQ2_VG027/report.json" | cut -d" " -f1)" = \
    08c78d66a90860cd8fdd7790e211b381969d9652943e74060c2692e3bf7eec8c
test "$(sha256sum "$VQ2_VG027/metadata.json" | cut -d" " -f1)" = \
    cf1e9ba23f9cb9e6da68226682c74b5cd51a57de3f478b9e09ea266eae9c6fb5
test "$(sha256sum "$VQ2_VG027_ADMISSION" | cut -d" " -f1)" = \
    de2437f3e44cecbb73c1d6b948f0218818d77f01570acd517c1f52fd3a755301

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG028 cannot see CUDA")
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
)"

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
    tests/test_vq2_recurrent_phase.py \
    tests/test_train_vq2_recurrent_bc.py \
    tests/test_train_vq2_variable_gate_recurrent_bc.py \
    tests/test_train_vq2_variable_gate_dagger_refit.py \
    tests/test_train_vq2_variable_gate_three_source_refit.py \
    tests/test_train_vq2_variable_gate_four_source_refit.py \
    tests/test_train_vq2_variable_gate_five_source_refit.py \
    tests/test_train_vq2_variable_gate_six_source_refit.py \
    tests/test_train_vq2_variable_gate_seven_source_refit.py

run_training
