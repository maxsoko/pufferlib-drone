#!/usr/bin/env bash
# Source-locked Vast bootstrap for the parity-gated VG022 continuation.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG022 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg022_five_source_device_decode_continuation_001}"
VQ2_STATE="$VQ2_OUTPUT/training_state.pt"
VQ2_MIGRATION_STATE="${VQ2_MIGRATION_STATE:-/workspace/vq2_vg020_epoch3_training_state.pt}"
VQ2_PARITY_REPORT="${VQ2_PARITY_REPORT:-/workspace/vq2_vg022_device_decode_parity_report.json}"

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
    echo "VG022 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

run_training() {
    if [ -f "$VQ2_STATE" ]; then
        python scripts/train_vq2_variable_gate_five_source_refit.py \
            --output "$VQ2_OUTPUT" \
            --device cuda \
            --parity-report "$VQ2_PARITY_REPORT" \
            --resume
    else
        python scripts/train_vq2_variable_gate_five_source_refit.py \
            --output "$VQ2_OUTPUT" \
            --device cuda \
            --migration-state "$VQ2_MIGRATION_STATE" \
            --parity-report "$VQ2_PARITY_REPORT"
    fi
}

# A persisted VG022 state is already source/runtime/parity-bound. Resume it
# without installing, rebuilding, retesting, or touching epoch-zero inputs.
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
test "$(sha256sum "$VQ2_MIGRATION_STATE" | cut -d" " -f1)" = \
    3d4694baaeac55184d3672675b4c159ee339adbff45c4b46246a58118ecbb14b

python - "$VQ2_MIGRATION_STATE" <<'PY'
import sys
import torch
from scripts.train_vq2_variable_gate_three_source_refit import runtime_manifest
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected = state["runtime"]
actual = runtime_manifest()
if actual != expected:
    raise SystemExit(f"VG022 runtime mismatch: {actual} != {expected}")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
    raise SystemExit(f"VG022 GPU mismatch: {torch.cuda.get_device_name(0)}")
print(actual)
PY

scripts/test_drone_race_native_regressions.sh
scripts/test_drone_race_vision_native_regressions.sh
VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
CUDA_HOME="$VQ2_CUDA_HOME" NVCC_ARCH=sm_89 \
    bash build.sh drone_race_vision --float
python - <<'PY'
from pufferlib import _C
assert _C.env_name == "drone_race_vision", _C.env_name
assert _C.precision_bytes == 4, _C.precision_bytes
PY

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python -m pytest -q \
    tests/test_vq2_recurrent_phase.py \
    tests/test_train_vq2_recurrent_bc.py \
    tests/test_train_vq2_variable_gate_recurrent_bc.py \
    tests/test_train_vq2_variable_gate_dagger_refit.py \
    tests/test_train_vq2_variable_gate_three_source_refit.py \
    tests/test_train_vq2_variable_gate_four_source_refit.py \
    tests/test_train_vq2_variable_gate_five_source_refit.py

python scripts/check_vq2_vg022_device_decode_parity.py \
    --migration-state "$VQ2_MIGRATION_STATE" \
    --output "$VQ2_PARITY_REPORT"
python - <<'PY' "$VQ2_PARITY_REPORT"
import json
import sys
report = json.load(open(sys.argv[1]))
if not report.get("admitted") or report.get("optimizer_steps") != 0:
    raise SystemExit("VG022 parity did not admit")
PY

run_training
