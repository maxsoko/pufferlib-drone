#!/usr/bin/env bash
set -euo pipefail
VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed LC057 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc057_grouped_single_vector_bracket_001}"
cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
source .venv/bin/activate
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_eval_vq2_lc057_grouped_single_vector_bracket.py \
    tests/test_eval_vq2_lc056_single_context_concurrent_bracket.py \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase_residual.py
bash build.sh drone_race_vision --float
OMP_NUM_THREADS=128 OMP_DYNAMIC=FALSE MKL_NUM_THREADS=1 \
    python scripts/eval_vq2_lc057_grouped_single_vector_bracket.py \
    --output "$VQ2_OUTPUT" --device cuda
