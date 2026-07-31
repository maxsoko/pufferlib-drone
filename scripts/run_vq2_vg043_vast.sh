#!/usr/bin/env bash
# Source-locked paired count-5 diagnostic for VG033 versus VG042.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set the pushed VG043 commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg043_vg042_paired_count5_diagnostic_001}"
VQ2_PARENT_OUTPUT="$VQ2_OUTPUT/parent"
VQ2_CANDIDATE_OUTPUT="$VQ2_OUTPUT/candidate"
VQ2_REPORT="$VQ2_OUTPUT/report.json"
VQ2_PARENT_MANIFEST="$VQ2_WORKSPACE_PATH/docs/vq2_vg043_parent_count5_manifest_2026-07-31.json"
VQ2_CANDIDATE_MANIFEST="$VQ2_WORKSPACE_PATH/docs/vq2_vg043_candidate_count5_manifest_2026-07-31.json"
VQ2_PREREGISTRATION="$VQ2_WORKSPACE_PATH/docs/vq2_vg043_paired_count5_diagnostic_preregistration_2026-07-31.md"

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
    echo "VG043 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

run_component() {
    local manifest="$1"
    local output="$2"
    local -a resume=()
    if [ -f "$output/state.json" ]; then
        resume=(--resume)
    fi
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
        python scripts/eval_vq2_staged_count5_component.py \
            --manifest "$manifest" \
            --output "$output" \
            --device cuda \
            "${resume[@]}"
}

run_comparator() {
    local -a resume=()
    if [ -f "$VQ2_REPORT" ]; then
        resume=(--resume)
    fi
    python scripts/compare_vq2_staged_count5_diagnostic.py \
        --parent-output "$VQ2_PARENT_OUTPUT" \
        --candidate-output "$VQ2_CANDIDATE_OUTPUT" \
        --preregistration "$VQ2_PREREGISTRATION" \
        --output "$VQ2_REPORT" \
        --num-gates 5 \
        "${resume[@]}"
}

if [ ! -f "$VQ2_PARENT_OUTPUT/state.json" ] \
        && [ ! -f "$VQ2_CANDIDATE_OUTPUT/state.json" ]; then
    for command_name in git clang ccache nvcc nvidia-smi python; do
        if ! command -v "$command_name" >/dev/null 2>&1; then
            echo "missing required command: $command_name" >&2
            exit 2
        fi
    done
    printf '#include <omp.h>\n' | clang -fopenmp -x c - -fsyntax-only
    if [ "$(nproc)" -lt 32 ]; then
        echo "VG043 requires at least 32 visible CPUs" >&2
        exit 2
    fi
    VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
    VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
    if [ "$VQ2_RAM_GIB" -lt 60 ]; then
        echo "VG043 requires approximately 64 GiB RAM" >&2
        exit 2
    fi
    if [ "$VQ2_DISK_FREE_GIB" -lt 15 ]; then
        echo "VG043 requires at least 15 GiB free" >&2
        exit 2
    fi

    test "$(sha256sum "$VQ2_PARENT_MANIFEST" | cut -d" " -f1)" = \
        ecb6dc3601dafe13eb1007f84c5dd517ecce8672df3186ed4e7e5999c56990be
    test "$(sha256sum "$VQ2_CANDIDATE_MANIFEST" | cut -d" " -f1)" = \
        c74dd150391f9efe0f8097860659c75d2cf2572aa13b7c08d1681cb4628c6584

    VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG043 cannot see CUDA")
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
        tests/test_vq2_staged_count5_diagnostic.py \
        tests/test_vq2_vg043_staged_count5_diagnostic.py \
        tests/test_eval_vq2_variable_gate_recurrent_policy.py \
        tests/test_eval_vq2_vg026_variable_gate_recurrent_policy.py \
        tests/test_eval_vq2_variable_gate_oracle.py \
        tests/test_eval_vq2_recurrent_policy.py \
        tests/test_vq2_public_phase.py \
        tests/test_vq2_recurrent_phase.py
fi

mkdir -p "$VQ2_OUTPUT"
set +e
run_component "$VQ2_PARENT_MANIFEST" "$VQ2_PARENT_OUTPUT" \
    >> "$VQ2_OUTPUT/parent_component.log" 2>&1 &
parent_pid=$!
run_component "$VQ2_CANDIDATE_MANIFEST" "$VQ2_CANDIDATE_OUTPUT" \
    >> "$VQ2_OUTPUT/candidate_component.log" 2>&1 &
candidate_pid=$!
wait "$parent_pid"
parent_exit=$?
wait "$candidate_pid"
candidate_exit=$?
set -e
printf '%s\n' "$parent_exit" > "$VQ2_OUTPUT/parent_component.exit"
printf '%s\n' "$candidate_exit" > "$VQ2_OUTPUT/candidate_component.exit"
if [ "$parent_exit" -ne 0 ] || [ "$candidate_exit" -ne 0 ]; then
    echo "VG043 component failed before paired comparison" >&2
    exit 2
fi

run_comparator
