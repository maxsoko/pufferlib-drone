#!/usr/bin/env bash
# Source-locked Vast bootstrap and resumable VG032 higher-phase collection.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG032 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg032_variable_gate_dagger_round7_vg028_visited_512}"
VQ2_STATE="${VQ2_OUTPUT}_state.json"
VQ2_MANIFEST="$VQ2_WORKSPACE_PATH/docs/vq2_vg032_variable_gate_dagger_manifest_2026-07-31.json"
VQ2_CANDIDATE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg028_variable_gate_seven_source_refit_001"
VQ2_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg028_seven_source_refit_admission_2026-07-31.json"
VQ2_SCREEN_EVIDENCE="$VQ2_WORKSPACE_PATH/docs/vq2_vg031_variable_gate_teacher_free_early_rejection_2026-07-31.json"
VQ2_ORACLE_REPORT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_sf016_alignment_oracle_query_parity_64/report.json"
VQ2_GOAL="$VQ2_WORKSPACE_PATH/docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"

run_vg032_collection() {
    python scripts/collect_vq2_staged_variable_gate_dagger.py \
        --manifest "$VQ2_MANIFEST" \
        --output "$VQ2_OUTPUT" \
        --device cuda \
        "$@"
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
    echo "VG032 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

if [ -f "$VQ2_STATE" ]; then
    run_vg032_collection --resume
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
    echo "VG032 requires at least 32 visible CPUs" >&2
    exit 2
fi
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG032 requires approximately 64 GiB RAM" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 20 ]; then
    echo "VG032 requires at least 20 GiB free" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_MANIFEST" | cut -d" " -f1)" = \
    d6da45a8cf38d8da2f47c8c025e81d23aee7b7dfec36e332fc6ef485c27b8c8d
test "$(sha256sum "$VQ2_CANDIDATE/policy_best.pt" | cut -d" " -f1)" = \
    ec1206b5dbccb2ee679fa049534956b189b419da2b927f5e3a61eeaf5a359e65
test "$(sha256sum "$VQ2_CANDIDATE/report.json" | cut -d" " -f1)" = \
    b51b7a9b25bc9f213a8b93a3d60fca6a2226d2511f9475c0a6da7e12cd21909e
test "$(sha256sum "$VQ2_ADMISSION" | cut -d" " -f1)" = \
    3282c98fab562a9593261538f452226190bc51b10e5e64495957cd348403755b
test "$(sha256sum "$VQ2_SCREEN_EVIDENCE" | cut -d" " -f1)" = \
    aed9cc0651a204fe4d29ddbd6ea34d7d376865f38ae0d0aca1160efbaea1f244
test "$(sha256sum "$VQ2_ORACLE_REPORT" | cut -d" " -f1)" = \
    a368916dadf058bbb0ede21838d8de7db4b2b1bba209d3e9ac78ccd3ea6e301d
test "$(sha256sum "$VQ2_GOAL" | cut -d" " -f1)" = \
    03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG032 cannot see CUDA")
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
    tests/test_collect_vq2_staged_variable_gate_dagger.py \
    tests/test_collect_vq2_variable_gate_oracle_bc_dataset.py \
    tests/test_eval_vq2_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg031_variable_gate_recurrent_policy.py \
    tests/test_summarize_vq2_variable_gate_screen.py \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase.py

run_vg032_collection
