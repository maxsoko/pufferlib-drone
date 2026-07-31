#!/usr/bin/env bash
# Execute the source-locked LC005 import repair on the retained Vast host.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT_ROOT="${VQ2_OUTPUT_ROOT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc005_scaleout_puffer_import_repair_001}"
VQ2_BASELINE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc001_long_course_oracle_001/baseline_20g.json"

cd "$VQ2_WORKSPACE_PATH"
if [ "$(git rev-parse HEAD)" != "$VQ2_EXPECTED_COMMIT" ]; then
    echo "remote HEAD does not match VQ2_EXPECTED_COMMIT" >&2
    exit 2
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "remote tracked worktree is dirty" >&2
    exit 2
fi
if [ -e "$VQ2_OUTPUT_ROOT" ]; then
    echo "refusing to overwrite LC005 evidence" >&2
    exit 2
fi
test "$(sha256sum "$VQ2_BASELINE" | cut -d' ' -f1)" = \
    "877b1d3086d9ed68fc17f10f31f58221492dde9a0183561e0e0b43d8f0a432e1"

source .venv/bin/activate
VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
)"
scripts/test_drone_race_native_regressions.sh
scripts/test_drone_race_vision_native_regressions.sh
CUDA_HOME="$VQ2_CUDA_HOME" NVCC_ARCH="$VQ2_NVCC_ARCH" \
    bash build.sh drone_race_vision --float
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_benchmark_vq2_lc004_scaleout_puffer_training.py \
    tests/test_benchmark_vq2_lc005_scaleout_puffer_import_repair.py

mkdir -p "$VQ2_OUTPUT_ROOT"
OMP_DYNAMIC=FALSE python \
    scripts/benchmark_vq2_lc005_scaleout_puffer_import_repair.py \
    --baseline "$VQ2_BASELINE" \
    --output "$VQ2_OUTPUT_ROOT/report.json" \
    >"$VQ2_OUTPUT_ROOT/stdout.log"
