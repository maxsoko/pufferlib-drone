#!/usr/bin/env bash
# Source-locked Vast.ai bootstrap and resumable VG024 DAgger collection.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT to the pushed source commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg024_variable_gate_dagger_round5_vg022_visited_512}"
VQ2_STATE="${VQ2_OUTPUT}_state.json"
VQ2_CANDIDATE="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg022_five_source_device_decode_continuation_001"
VQ2_SCREEN="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg023_variable_gate_recurrent_teacher_free_256"
VQ2_ADMISSION="$VQ2_WORKSPACE_PATH/docs/vq2_vg022_five_source_device_decode_admission_2026-07-30.json"
VQ2_REJECTION="$VQ2_WORKSPACE_PATH/docs/vq2_vg023_variable_gate_teacher_free_rejection_2026-07-30.json"

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
    echo "VG024 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

if [ -f "$VQ2_STATE" ]; then
    python scripts/collect_vq2_vg024_variable_gate_dagger.py \
        --output "$VQ2_OUTPUT" \
        --device cuda \
        --resume
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
    echo "VG024 requires at least 32 visible CPUs" >&2
    exit 2
fi
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG024 requires approximately 64 GiB RAM" >&2
    exit 2
fi
if [ "$VQ2_DISK_FREE_GIB" -lt 15 ]; then
    echo "VG024 requires at least 15 GiB free" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_CANDIDATE/policy_best.pt" | cut -d" " -f1)" = \
    bd3f93d46e2b04af4a1ae843ac8a920bd75e2954079dbba3c6a3da541bb93915
test "$(sha256sum "$VQ2_CANDIDATE/report.json" | cut -d" " -f1)" = \
    23d281824813e14c3d77c4446a36f4af4d29769c01bdd0a46202caae017c95b3
test "$(sha256sum "$VQ2_ADMISSION" | cut -d" " -f1)" = \
    ac93a00d46139a97610731acac37f6b652be69c7c2d8aa30d865c1a97c593994
test "$(sha256sum "$VQ2_SCREEN/report.json" | cut -d" " -f1)" = \
    2a3023fdedf080e6272e74b88e61d5b5e23f6395cfc3b2c1cd40d049c5504665
test "$(sha256sum "$VQ2_REJECTION" | cut -d" " -f1)" = \
    88f403f028eece863e6971e12d6d0fae675cf3e4aec6b85f7b5cb75274329d34

VQ2_CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
VQ2_NVCC_ARCH="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("VG024 cannot see CUDA")
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
    tests/test_collect_vq2_vg019_variable_gate_dagger.py \
    tests/test_collect_vq2_vg024_variable_gate_dagger.py \
    tests/test_collect_vq2_variable_gate_oracle_bc_dataset.py \
    tests/test_eval_vq2_variable_gate_recurrent_policy.py \
    tests/test_eval_vq2_vg023_variable_gate_recurrent_policy.py \
    tests/test_vq2_public_phase.py \
    tests/test_vq2_recurrent_phase.py

python scripts/collect_vq2_vg024_variable_gate_dagger.py \
    --output "$VQ2_OUTPUT" \
    --device cuda
