#!/usr/bin/env bash
# Source-locked Vast bootstrap and resumable VG063 corrected intervention.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG063 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg063_horizon_corrected_intervention_features_001}"
VQ2_VG062_REJECTION="$VQ2_WORKSPACE_PATH/docs/vq2_vg062_warmed_teacher_intervention_rejection_2026-07-31.json"

run_collection() {
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/collect_vq2_vg063_horizon_corrected_intervention_features.py \
            --output "$VQ2_OUTPUT" --device cuda "$@"
}

cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test -x .venv/bin/python
source .venv/bin/activate

if [ -f "${VQ2_OUTPUT}_state.json" ]; then
    run_collection --resume
    exit $?
fi

for command_name in git clang ccache nvcc nvidia-smi python; do
    command -v "$command_name" >/dev/null
done
test "$(nproc)" -ge 32
test "$(df --output=avail -B1 "$VQ2_WORKSPACE_PATH" | tail -1)" -ge 20000000000
test "$(sha256sum "$VQ2_VG062_REJECTION" | cut -d" " -f1)" = \
    397b614af2bf3aac3d8ef6e9a1dd2180a08af865d17b53d0685d4d309916ce3f

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
)"
scripts/test_drone_race_native_regressions.sh
scripts/test_drone_race_vision_native_regressions.sh
CUDA_HOME="$VQ2_CUDA_HOME" NVCC_ARCH="$VQ2_NVCC_ARCH" \
    bash build.sh drone_race_vision --float
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_collect_vq2_vg062_warmed_teacher_intervention_features.py \
    tests/test_collect_vq2_vg063_horizon_corrected_intervention_features.py \
    tests/test_collect_vq2_variable_gate_dagger.py \
    tests/test_eval_vq2_variable_gate_recurrent_policy.py
run_collection
