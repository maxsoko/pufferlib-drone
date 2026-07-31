#!/usr/bin/env bash
set -euo pipefail
VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VG051 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg051_vg033_count6_multi_offset_baseline_001}"
cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
source .venv/bin/activate
resume=()
[ ! -f "$VQ2_OUTPUT/state.json" ] || resume=(--resume)
if [ "${#resume[@]}" -eq 0 ]; then
    scripts/test_drone_race_native_regressions.sh
    scripts/test_drone_race_vision_native_regressions.sh
    VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
)"
    CUDA_HOME="$VQ2_CUDA_HOME" NVCC_ARCH="$VQ2_NVCC_ARCH" \
        bash build.sh drone_race_vision --float
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 python -m pytest -q \
        tests/test_eval_vq2_vg051_count6_multi_offset_baseline.py \
        tests/test_vq2_staged_count5_diagnostic.py \
        tests/test_eval_vq2_variable_gate_recurrent_policy.py
fi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
    python scripts/eval_vq2_vg051_count6_multi_offset_baseline.py \
    --output "$VQ2_OUTPUT" --device cuda "${resume[@]}"
