#!/usr/bin/env bash
# Source-locked Vast bootstrap and bounded-thread VG037 teacher-free screen.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG037 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg037_vg033_variable_gate_recurrent_teacher_free_256}"
VQ2_STATE="$VQ2_OUTPUT/state.json"
VQ2_CANDIDATE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg033_variable_gate_eight_source_refit_001"
VQ2_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
VQ2_DIAGNOSTIC="$VQ2_WORKSPACE_PATH/docs/vq2_vg035_paired_count11_diagnostic_admission_2026-07-31.json"
VQ2_ABORT="$VQ2_WORKSPACE_PATH/docs/vq2_vg036_infrastructure_abort_2026-07-31.json"
VQ2_GOAL="$VQ2_WORKSPACE_PATH/docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"

run_vg037_screen() {
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/eval_vq2_vg037_variable_gate_recurrent_policy.py \
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
    echo "VG037 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

if [ -f "$VQ2_STATE" ]; then
    run_vg037_screen
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
    echo "VG037 requires at least 32 visible CPUs" >&2
    exit 2
fi
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG037 requires approximately 64 GiB RAM" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 15 ]; then
    echo "VG037 requires at least 15 GiB free" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_CANDIDATE/policy_best.pt" | cut -d" " -f1)" = \
    56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26
test "$(sha256sum "$VQ2_CANDIDATE/report.json" | cut -d" " -f1)" = \
    033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60
test "$(sha256sum "$VQ2_ADMISSION" | cut -d" " -f1)" = \
    dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44
test "$(sha256sum "$VQ2_DIAGNOSTIC" | cut -d" " -f1)" = \
    7c7676b72429157c8434761bf39ec790eb00a53abb76b4733128b980d8d95d98
test "$(sha256sum "$VQ2_ABORT" | cut -d" " -f1)" = \
    c5dc3287c2891c746de4eadd55df8648df7762b45e2655baeff623a29af7c27e
test "$(sha256sum "$VQ2_GOAL" | cut -d" " -f1)" = \
    03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG037 cannot see CUDA")
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
    tests/test_eval_vq2_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg018_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg023_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg026_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg031_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg036_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg037_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_variable_gate_oracle.py \
    tests/test_eval_vq2_recurrent_policy.py \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase.py

OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
    python scripts/eval_vq2_vg037_variable_gate_recurrent_policy.py \
        --output "$VQ2_OUTPUT" \
        --device cuda
