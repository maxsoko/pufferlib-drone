#!/usr/bin/env bash
# Build and run source-locked long-course teacher/throughput proofs on Vast.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT_ROOT="${VQ2_OUTPUT_ROOT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc001_long_course_oracle_001}"

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
    echo "refusing to overwrite LC001 evidence" >&2
    exit 2
fi

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
python -m pytest -q \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase_residual.py \
    tests/test_eval_vq2_lc001_long_course_oracle.py

mkdir -p "$VQ2_OUTPUT_ROOT"
python scripts/eval_vq2_lc001_long_course_oracle.py \
    --count 20 --agents 1 --episodes 1 --threads 1 --seed 431001 \
    --output "$VQ2_OUTPUT_ROOT/baseline_20g.json" \
    >"$VQ2_OUTPUT_ROOT/baseline_20g.stdout"

python scripts/eval_vq2_lc001_long_course_oracle.py \
    --count 20 --agents 64 --episodes 64 --threads 32 --seed 431002 \
    --output "$VQ2_OUTPUT_ROOT/vector_20g.json" \
    >"$VQ2_OUTPUT_ROOT/vector_20g.stdout" &
VQ2_PID_20=$!
python scripts/eval_vq2_lc001_long_course_oracle.py \
    --count 24 --agents 64 --episodes 64 --threads 32 --seed 431003 \
    --output "$VQ2_OUTPUT_ROOT/vector_24g.json" \
    >"$VQ2_OUTPUT_ROOT/vector_24g.stdout" &
VQ2_PID_24=$!

wait "$VQ2_PID_20"
wait "$VQ2_PID_24"
python scripts/eval_vq2_lc001_long_course_oracle.py --aggregate \
    --baseline "$VQ2_OUTPUT_ROOT/baseline_20g.json" \
    --vector20 "$VQ2_OUTPUT_ROOT/vector_20g.json" \
    --vector24 "$VQ2_OUTPUT_ROOT/vector_24g.json" \
    --output "$VQ2_OUTPUT_ROOT/report.json"
