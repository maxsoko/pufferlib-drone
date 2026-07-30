#!/usr/bin/env bash
# Source-locked Vast.ai bootstrap and resumable VG018 teacher-free screen.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT to the pushed source commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg018_variable_gate_recurrent_teacher_free_256}"
VQ2_STATE="$VQ2_OUTPUT/state.json"
VQ2_CANDIDATE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg017_variable_gate_four_source_refit_001"
VQ2_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg017_four_source_refit_admission_2026-07-30.json"

run_vg018_screen() {
    python scripts/eval_vq2_vg018_variable_gate_recurrent_policy.py \
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
    echo "VG018 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

if [ -f "$VQ2_STATE" ]; then
    run_vg018_screen
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
    echo "VG018 requires at least 32 visible CPUs" >&2
    exit 2
fi
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG018 requires approximately 64 GiB RAM" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 15 ]; then
    echo "VG018 requires at least 15 GiB free" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_CANDIDATE/policy_best.pt" | cut -d" " -f1)" = \
    6769476f309beb4bb502c64a83c4f992b47da2b1a8d8b6913398b4b19b3ab4a8
test "$(sha256sum "$VQ2_CANDIDATE/report.json" | cut -d" " -f1)" = \
    476c730649c0f127feb4975fe2ad44f66be64b0ea8dde4d48c5ada7bb0d4c86b
test "$(sha256sum "$VQ2_ADMISSION" | cut -d" " -f1)" = \
    df664cbac28ee676cbd5e056cb1f06745bf3e622710b3faa085eca90425741b6

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG018 cannot see CUDA")
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
)"

python - <<'PY'
import torch
name = torch.cuda.get_device_name(0)
if not any(candidate in name for candidate in ("RTX 4090", "A100", "RTX A6000")):
    raise SystemExit(f"unsupported VG018 GPU: {name}")
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
    tests/test_eval_vq2_vg018_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_variable_gate_oracle.py \
    tests/test_eval_vq2_recurrent_policy.py \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase.py

python scripts/eval_vq2_vg018_variable_gate_recurrent_policy.py \
    --output "$VQ2_OUTPUT" \
    --device cuda
