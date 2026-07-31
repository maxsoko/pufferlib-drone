#!/usr/bin/env bash
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG050 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg050_vg049_offset24_paired_count5_confirmation_001}"
VQ2_PARENT_OUTPUT="$VQ2_OUTPUT/parent"
VQ2_CANDIDATE_OUTPUT="$VQ2_OUTPUT/candidate"
VQ2_REPORT="$VQ2_OUTPUT/report.json"
VQ2_PARENT_MANIFEST="$VQ2_WORKSPACE_PATH/docs/vq2_vg050_parent_count5_manifest_2026-07-31.json"
VQ2_CANDIDATE_MANIFEST="$VQ2_WORKSPACE_PATH/docs/vq2_vg050_candidate_count5_manifest_2026-07-31.json"
VQ2_PREREGISTRATION="$VQ2_WORKSPACE_PATH/docs/vq2_vg050_offset24_paired_count5_confirmation_preregistration_2026-07-31.md"

cd "$VQ2_WORKSPACE_PATH"
test "$(git rev-parse HEAD)" = "$VQ2_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
source .venv/bin/activate

run_component() {
    local manifest="$1"
    local output="$2"
    local -a resume=()
    [ ! -f "$output/state.json" ] || resume=(--resume)
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/eval_vq2_staged_count5_component.py \
        --manifest "$manifest" --output "$output" --device cuda "${resume[@]}"
}

run_comparator() {
    local -a resume=()
    [ ! -f "$VQ2_REPORT" ] || resume=(--resume)
    python scripts/compare_vq2_vg050_offset24_paired_count5.py \
        --parent-output "$VQ2_PARENT_OUTPUT" \
        --candidate-output "$VQ2_CANDIDATE_OUTPUT" \
        --preregistration "$VQ2_PREREGISTRATION" \
        --output "$VQ2_REPORT" "${resume[@]}"
}

if [ ! -f "$VQ2_PARENT_OUTPUT/state.json" ] \
        && [ ! -f "$VQ2_CANDIDATE_OUTPUT/state.json" ]; then
    test "$(sha256sum "$VQ2_PARENT_MANIFEST" | cut -d" " -f1)" = \
        843bea12f582c434810f147f77d5487a8628d0ba122b3b05a3634b8a81a9ea39
    test "$(sha256sum "$VQ2_CANDIDATE_MANIFEST" | cut -d" " -f1)" = \
        d2e08008d69dcf7d090fdfe4a26687c5bb5f51b6a26d21fd983b99404c175e39
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
        tests/test_vq2_vg050_offset24_paired_count5.py \
        tests/test_vq2_staged_count5_diagnostic.py \
        tests/test_eval_vq2_variable_gate_recurrent_policy.py
fi

mkdir -p "$VQ2_OUTPUT"
set +e
run_component "$VQ2_PARENT_MANIFEST" "$VQ2_PARENT_OUTPUT" \
    > "$VQ2_OUTPUT/parent_component.log" 2>&1 &
parent_pid=$!
run_component "$VQ2_CANDIDATE_MANIFEST" "$VQ2_CANDIDATE_OUTPUT" \
    > "$VQ2_OUTPUT/candidate_component.log" 2>&1 &
candidate_pid=$!
wait "$parent_pid"; parent_exit=$?
wait "$candidate_pid"; candidate_exit=$?
set -e
printf '%s\n' "$parent_exit" > "$VQ2_OUTPUT/parent_component.exit"
printf '%s\n' "$candidate_exit" > "$VQ2_OUTPUT/candidate_component.exit"
test "$parent_exit" -eq 0
test "$candidate_exit" -eq 0
run_comparator
