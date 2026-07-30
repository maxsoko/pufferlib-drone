#!/usr/bin/env bash
# Source-locked Vast bootstrap and epoch-resumable VG025 six-source refit.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG025 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg025_variable_gate_six_source_refit_001}"
VQ2_STATE="$VQ2_OUTPUT/training_state.pt"
VQ2_VG024="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg024_variable_gate_dagger_round5_vg022_visited_512"
VQ2_VG024_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg024_variable_gate_dagger_round5_admission_2026-07-30.json"
VQ2_PARENT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg022_five_source_device_decode_continuation_001"

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
    echo "VG025 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

run_training() {
    if [ -f "$VQ2_STATE" ]; then
        python scripts/train_vq2_variable_gate_six_source_refit.py \
            --output "$VQ2_OUTPUT" \
            --device cuda \
            --resume
    else
        python scripts/train_vq2_variable_gate_six_source_refit.py \
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
    echo "VG025 requires at least 32 visible CPUs" >&2
    exit 2
fi
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG025 requires approximately 64 GiB RAM" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 15 ]; then
    echo "VG025 requires at least 15 GiB free" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_PARENT/policy_best.pt" | cut -d" " -f1)" = \
    bd3f93d46e2b04af4a1ae843ac8a920bd75e2954079dbba3c6a3da541bb93915
test "$(sha256sum "$VQ2_PARENT/report.json" | cut -d" " -f1)" = \
    23d281824813e14c3d77c4446a36f4af4d29769c01bdd0a46202caae017c95b3
test "$(sha256sum "$VQ2_VG024/report.json" | cut -d" " -f1)" = \
    5d36213f14e8efca977fc9cfb501d9e264ec8e1856fcd4a5c71bc0a45d6eaa96
test "$(sha256sum "$VQ2_VG024/metadata.json" | cut -d" " -f1)" = \
    dfe6474521a063f923d7de1d7303cc3004087e1a115b2fc713353be1ac2c69c4
test "$(sha256sum "$VQ2_VG024_ADMISSION" | cut -d" " -f1)" = \
    3b31cb2e3d5d60d0443cb46eaa3e9facc0b9673ab233e253a1de20e506e201dd

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG025 cannot see CUDA")
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
    tests/test_train_vq2_variable_gate_six_source_refit.py

run_training
