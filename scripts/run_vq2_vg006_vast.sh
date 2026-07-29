#!/usr/bin/env bash
# Source-locked Vast.ai bootstrap and resumable VG006 teacher-free screen.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT to the pushed source commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg006_variable_gate_recurrent_teacher_free_256}"
VQ2_STATE="$VQ2_OUTPUT/state.json"
VQ2_CHECKPOINT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg005_variable_gate_recurrent_bc_001/policy_best.pt"
VQ2_TRAIN_REPORT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg005_variable_gate_recurrent_bc_001/report.json"
VQ2_CHECKPOINT_SHA256="f686a35dc923137bc40fd6feca459e52e5585d5b6fa047b7f77a171e2832a6c3"
VQ2_TRAIN_REPORT_SHA256="bdcd2b38ea3537f2c250281c63f2e6aeca0ad87c97521339f098eec7ff474e5b"

run_vg006_screen() {
    python scripts/eval_vq2_variable_gate_recurrent_policy.py \
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
    echo "VG006 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

# Resume only the exact screen state. Rebuilding after a state exists would
# change the compiled-extension source identity.
if [ -f "$VQ2_STATE" ]; then
    run_vg006_screen
    exit $?
fi

for command_name in git clang nvcc nvidia-smi python; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "missing required command: $command_name" >&2
        exit 2
    fi
done
if [ "$(nproc)" -lt 32 ]; then
    echo "VG006 requires at least 32 visible CPUs" >&2
    exit 2
fi
if [ -z "$VQ2_CHECKPOINT_SHA256" ] || [ -z "$VQ2_TRAIN_REPORT_SHA256" ]; then
    echo "VG006 checkpoint/report hashes were not source-locked" >&2
    exit 2
fi
test "$(sha256sum "$VQ2_CHECKPOINT" | cut -d" " -f1)" = "$VQ2_CHECKPOINT_SHA256"
test "$(sha256sum "$VQ2_TRAIN_REPORT" | cut -d" " -f1)" = "$VQ2_TRAIN_REPORT_SHA256"

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG006 cannot see CUDA")
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

python -m pytest -q \
    tests/test_eval_vq2_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_variable_gate_oracle.py \
    tests/test_eval_vq2_recurrent_policy.py \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase.py

python scripts/eval_vq2_variable_gate_recurrent_policy.py \
    --output "$VQ2_OUTPUT" \
    --device cuda
