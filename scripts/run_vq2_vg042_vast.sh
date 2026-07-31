#!/usr/bin/env bash
# Source-locked Vast bootstrap and epoch-resumable VG042 phase-balanced refit.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG042 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg042_variable_gate_phase_balanced_refit_001}"
VQ2_STATE="$VQ2_OUTPUT/training_state.pt"
VQ2_VG039="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg039_variable_gate_dagger_round9_vg033_visited_512"
VQ2_VG039_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg039_variable_gate_dagger_round9_admission_2026-07-31.json"
VQ2_PARENT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg033_variable_gate_eight_source_refit_001"
VQ2_PARENT_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
VQ2_REJECTION="$VQ2_WORKSPACE_PATH/docs/vq2_vg041_paired_count5_diagnostic_rejection_2026-07-31.json"
VQ2_GOAL="$VQ2_WORKSPACE_PATH/docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"

run_training() {
    local -a resume=()
    if [ -f "$VQ2_STATE" ]; then
        resume=(--resume)
    fi
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/train_vq2_variable_gate_phase_balanced_refit.py \
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
    echo "VG042 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

# Epoch state binds sources, runtime, RNG, optimizer, and every corpus.
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
    echo "VG042 requires at least 32 visible CPUs" >&2
    exit 2
fi
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG042 requires approximately 64 GiB RAM" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 15 ]; then
    echo "VG042 requires at least 15 GiB free" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_PARENT/policy_best.pt" | cut -d" " -f1)" = \
    56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26
test "$(sha256sum "$VQ2_PARENT/report.json" | cut -d" " -f1)" = \
    033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60
test "$(sha256sum "$VQ2_PARENT_ADMISSION" | cut -d" " -f1)" = \
    dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44
test "$(sha256sum "$VQ2_VG039/report.json" | cut -d" " -f1)" = \
    a0bdcd0d60f66f7de55323a08b3c4695a0333b5bf88a93454d56d75e61eb21e1
test "$(sha256sum "$VQ2_VG039/metadata.json" | cut -d" " -f1)" = \
    89134752057b0dd7759031aaf524a93f0a85096bd1b8973ed6daebb5715c74ac
test "$(sha256sum "$VQ2_VG039_ADMISSION" | cut -d" " -f1)" = \
    e01eb2e28c65f827cfb39916c28d56320c57a4c23513ee61553447bd236b11cc
test "$(sha256sum "$VQ2_REJECTION" | cut -d" " -f1)" = \
    a9eb3079c7e59cf52594ce89c07182356724d99a0ab545c7e96eb9f954a55dd9
test "$(sha256sum "$VQ2_GOAL" | cut -d" " -f1)" = \
    03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG042 cannot see CUDA")
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
    tests/test_vq2_recurrent_phase.py \
    tests/test_train_vq2_recurrent_bc.py \
    tests/test_train_vq2_variable_gate_recurrent_bc.py \
    tests/test_train_vq2_variable_gate_dagger_refit.py \
    tests/test_train_vq2_variable_gate_three_source_refit.py \
    tests/test_train_vq2_variable_gate_four_source_refit.py \
    tests/test_train_vq2_variable_gate_five_source_refit.py \
    tests/test_train_vq2_variable_gate_six_source_refit.py \
    tests/test_train_vq2_variable_gate_seven_source_refit.py \
    tests/test_train_vq2_variable_gate_eight_source_refit.py \
    tests/test_train_vq2_variable_gate_nine_source_refit.py \
    tests/test_train_vq2_variable_gate_phase_balanced_refit.py

run_training
