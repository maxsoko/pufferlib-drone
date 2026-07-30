#!/usr/bin/env bash
# Source-locked Vast.ai bootstrap and resumable VG019 DAgger collection.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT to the pushed source commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg019_variable_gate_dagger_round4_vg017_visited_512}"
VQ2_STATE="${VQ2_OUTPUT}_state.json"
VQ2_CANDIDATE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg017_variable_gate_four_source_refit_001"
VQ2_SCREEN="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg018_variable_gate_recurrent_teacher_free_256"
VQ2_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg017_four_source_refit_admission_2026-07-30.json"
VQ2_REJECTION="$VQ2_WORKSPACE_PATH/docs/vq2_vg018_variable_gate_teacher_free_rejection_2026-07-30.json"

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
    echo "VG019 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

if [ -f "$VQ2_STATE" ]; then
    python scripts/collect_vq2_vg019_variable_gate_dagger.py \
        --output "$VQ2_OUTPUT" \
        --device cuda \
        --resume
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
    echo "VG019 requires at least 32 visible CPUs" >&2
    exit 2
fi
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG019 requires approximately 64 GiB RAM" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 15 ]; then
    echo "VG019 requires at least 15 GiB free" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_CANDIDATE/policy_best.pt" | cut -d" " -f1)" = \
    6769476f309beb4bb502c64a83c4f992b47da2b1a8d8b6913398b4b19b3ab4a8
test "$(sha256sum "$VQ2_CANDIDATE/report.json" | cut -d" " -f1)" = \
    476c730649c0f127feb4975fe2ad44f66be64b0ea8dde4d48c5ada7bb0d4c86b
test "$(sha256sum "$VQ2_ADMISSION" | cut -d" " -f1)" = \
    df664cbac28ee676cbd5e056cb1f06745bf3e622710b3faa085eca90425741b6
test "$(sha256sum "$VQ2_SCREEN/report.json" | cut -d" " -f1)" = \
    40be8bf0041e9b183c5ad057ae6b4e5a20eb82ef21a972a8f0390e9d3c65ca09
test "$(sha256sum "$VQ2_REJECTION" | cut -d" " -f1)" = \
    b0d7ddb33b887cdfeb840710915ebb07ff9ec8455d8e1890c1be722db6e8af36

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG019 cannot see CUDA")
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
    tests/test_collect_vq2_variable_gate_dagger.py \
    tests/test_collect_vq2_vg019_variable_gate_dagger.py \
    tests/test_collect_vq2_variable_gate_oracle_bc_dataset.py \
    tests/test_eval_vq2_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg018_variable_gate_recurrent_policy.py \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase.py

python scripts/collect_vq2_vg019_variable_gate_dagger.py \
    --output "$VQ2_OUTPUT" \
    --device cuda
