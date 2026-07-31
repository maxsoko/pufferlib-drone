#!/usr/bin/env bash
# Source-locked Vast runner for the VG047 offset-8 fraction bracket.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG047 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg047_vg033_to_vg042_offset8_fraction_bracket_001}"
VQ2_STATE="$VQ2_OUTPUT/state.json"
VQ2_PARENT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg033_variable_gate_eight_source_refit_001"
VQ2_UPDATE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg042_variable_gate_phase_balanced_refit_001"
VQ2_VG044="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg044_vg033_to_vg042_checkpoint_line_bracket_001"
VQ2_VG044_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg044_checkpoint_line_bracket_admission_2026-07-31.json"
VQ2_REJECTION="$VQ2_WORKSPACE_PATH/docs/vq2_vg046_offset8_paired_count5_rejection_2026-07-31.json"

run_bracket() {
    local -a resume=()
    if [ -f "$VQ2_STATE" ]; then
        resume=(--resume)
    fi
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/eval_vq2_vg047_offset8_fraction_bracket.py \
            --output "$VQ2_OUTPUT" \
            --device cuda \
            "${resume[@]}"
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
    echo "VG047 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

if [ -f "$VQ2_STATE" ]; then
    run_bracket
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
    echo "VG047 requires at least 32 visible CPUs" >&2
    exit 2
fi
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG047 requires approximately 64 GiB RAM" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 15 ]; then
    echo "VG047 requires at least 15 GiB free" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_PARENT/policy_best.pt" | cut -d" " -f1)" = \
    56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26
test "$(sha256sum "$VQ2_UPDATE/policy_best.pt" | cut -d" " -f1)" = \
    876b0ecc4dae769f6a9dd04a0114e9623961cf46397499a41d326bdc74141ddc
test "$(sha256sum "$VQ2_VG044/report.json" | cut -d" " -f1)" = \
    c6fb4544265342e2b33974196d25dba5dba4f18fd1039691f7f9ed2155b9a84f
test "$(sha256sum "$VQ2_VG044_ADMISSION" | cut -d" " -f1)" = \
    85f1baf2fb5c053ac9c0f908dc0170a43aae126eaf61a1e1ccf3110dc35c0fb1
test "$(sha256sum "$VQ2_REJECTION" | cut -d" " -f1)" = \
    3049a97ad216a90855a2b3ebeba199782390b09df87307d2f2d09e8bebf06055

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG047 cannot see CUDA")
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

OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_eval_vq2_vg047_offset8_fraction_bracket.py \
    tests/test_eval_vq2_vg044_checkpoint_line_bracket.py \
    tests/test_eval_vq2_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg026_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_variable_gate_oracle.py \
    tests/test_eval_vq2_recurrent_policy.py \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase.py

run_bracket
