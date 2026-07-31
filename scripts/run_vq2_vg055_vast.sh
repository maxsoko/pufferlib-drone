#!/usr/bin/env bash
# Source-locked Vast bootstrap and resumable VG055 fraction bracket.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG055 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg055_vg033_to_vg053_count6_fraction_bracket_001}"
VQ2_PARENT="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg033_variable_gate_eight_source_refit_001"
VQ2_UPDATE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg053_late_gate_trust_refit_001"
VQ2_PARENT_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
VQ2_UPDATE_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg053_late_gate_trust_refit_admission_2026-07-31.json"
VQ2_REJECTION="$VQ2_WORKSPACE_PATH/docs/vq2_vg054_count6_multi_offset_rejection_2026-07-31.json"
VQ2_GOAL="$VQ2_WORKSPACE_PATH/docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"

run_screen() {
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/eval_vq2_vg055_count6_fraction_bracket.py \
            --output "$VQ2_OUTPUT" --device cuda "$@"
}

cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test -x .venv/bin/python
source .venv/bin/activate

if [ -f "$VQ2_OUTPUT/state.json" ]; then
    run_screen --resume
    exit $?
fi

for command_name in git clang ccache nvcc nvidia-smi python; do
    command -v "$command_name" >/dev/null
done
test "$(sha256sum "$VQ2_PARENT/policy_best.pt" | cut -d" " -f1)" = \
    56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26
test "$(sha256sum "$VQ2_PARENT/report.json" | cut -d" " -f1)" = \
    033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60
test "$(sha256sum "$VQ2_PARENT_ADMISSION" | cut -d" " -f1)" = \
    dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44
test "$(sha256sum "$VQ2_UPDATE/policy_best.pt" | cut -d" " -f1)" = \
    f0a9812f55ce95d437af06a043c64d75a3ea757ac6965df9e3cdf64f30563ce5
test "$(sha256sum "$VQ2_UPDATE/report.json" | cut -d" " -f1)" = \
    ec8b81b3d357fcbfb4105ceda1fd22b43ce921d08b238bea35224fc49485f8b6
test "$(sha256sum "$VQ2_UPDATE_ADMISSION" | cut -d" " -f1)" = \
    3838729ec1a5d2ffbb596aae436804099ce361035eccce91b2ad83145770b134
test "$(sha256sum "$VQ2_REJECTION" | cut -d" " -f1)" = \
    bb78f928927d3b790e58cb17c79c88bcbc49a14ea4edaa27b229e059b8ddbbf4
test "$(sha256sum "$VQ2_GOAL" | cut -d" " -f1)" = \
    03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1

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
    tests/test_eval_vq2_vg055_count6_fraction_bracket.py \
    tests/test_eval_vq2_vg044_checkpoint_line_bracket.py \
    tests/test_eval_vq2_candidate_count6_multi_offset.py \
    tests/test_eval_vq2_variable_gate_recurrent_policy.py

run_screen
