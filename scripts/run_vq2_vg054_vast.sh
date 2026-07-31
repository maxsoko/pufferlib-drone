#!/usr/bin/env bash
# Source-locked Vast bootstrap and resumable VG054 six-gate screen.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG054 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg054_vg053_count6_multi_offset_001}"
VQ2_MANIFEST="$VQ2_WORKSPACE_PATH/docs/vq2_vg054_count6_multi_offset_manifest_2026-07-31.json"
VQ2_CANDIDATE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg053_late_gate_trust_refit_001"
VQ2_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg053_late_gate_trust_refit_admission_2026-07-31.json"
VQ2_BASELINE="$VQ2_WORKSPACE_PATH/docs/vq2_vg051_count6_multi_offset_baseline_evidence_2026-07-31.json"

run_screen() {
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/eval_vq2_candidate_count6_multi_offset.py \
            --manifest "$VQ2_MANIFEST" \
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
test "$(sha256sum "$VQ2_MANIFEST" | cut -d" " -f1)" = \
    001e385a1616f3fb5582e85e1bf594b76ef4b72721b690eaf609646d7502d419
test "$(sha256sum "$VQ2_CANDIDATE/policy_best.pt" | cut -d" " -f1)" = \
    f0a9812f55ce95d437af06a043c64d75a3ea757ac6965df9e3cdf64f30563ce5
test "$(sha256sum "$VQ2_CANDIDATE/report.json" | cut -d" " -f1)" = \
    ec8b81b3d357fcbfb4105ceda1fd22b43ce921d08b238bea35224fc49485f8b6
test "$(sha256sum "$VQ2_ADMISSION" | cut -d" " -f1)" = \
    3838729ec1a5d2ffbb596aae436804099ce361035eccce91b2ad83145770b134
test "$(sha256sum "$VQ2_BASELINE" | cut -d" " -f1)" = \
    88e56a5b85689c564a2bb26036a3243ded5efee9155af6ec0135a522c80247ed

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
    tests/test_eval_vq2_candidate_count6_multi_offset.py \
    tests/test_eval_vq2_vg051_count6_multi_offset_baseline.py \
    tests/test_vq2_staged_count5_diagnostic.py \
    tests/test_eval_vq2_variable_gate_recurrent_policy.py

run_screen
