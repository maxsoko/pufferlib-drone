#!/usr/bin/env bash
# Source-locked Vast runner for the VG049 offset-16 micro bracket.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG049 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg049_vg033_to_vg042_offset16_micro_fraction_bracket_001}"
VQ2_STATE="$VQ2_OUTPUT/state.json"
VQ2_PARENT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg033_variable_gate_eight_source_refit_001"
VQ2_UPDATE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg042_variable_gate_phase_balanced_refit_001"
VQ2_REJECTION="$VQ2_WORKSPACE_PATH/docs/vq2_vg048_offset16_paired_count5_rejection_2026-07-31.json"

run_bracket() {
    local -a resume=()
    if [ -f "$VQ2_STATE" ]; then
        resume=(--resume)
    fi
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/eval_vq2_vg049_offset16_micro_fraction_bracket.py \
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
source .venv/bin/activate

if [ -f "$VQ2_STATE" ]; then
    run_bracket
    exit $?
fi

for command_name in git clang ccache nvcc nvidia-smi python; do
    command -v "$command_name" >/dev/null 2>&1
done
printf '#include <omp.h>\n' | clang -fopenmp -x c - -fsyntax-only
test "$(nproc)" -ge 32
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
test "$VQ2_RAM_GIB" -ge 60
test "$VQ2_DISK_FREE_GIB" -ge 15
test "$(sha256sum "$VQ2_PARENT/policy_best.pt" | cut -d" " -f1)" = \
    56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26
test "$(sha256sum "$VQ2_UPDATE/policy_best.pt" | cut -d" " -f1)" = \
    876b0ecc4dae769f6a9dd04a0114e9623961cf46397499a41d326bdc74141ddc
test "$(sha256sum "$VQ2_REJECTION" | cut -d" " -f1)" = \
    a8f5b91f24d8713348e860149520300438175dfe5b7bb1d4f0877cb5e60bf440

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG049 cannot see CUDA")
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
assert _C.env_name == "drone_race_vision"
assert _C.precision_bytes == 4
PY
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_eval_vq2_vg049_offset16_micro_fraction_bracket.py \
    tests/test_eval_vq2_vg047_offset8_fraction_bracket.py \
    tests/test_eval_vq2_vg044_checkpoint_line_bracket.py \
    tests/test_eval_vq2_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_recurrent_policy.py

run_bracket
