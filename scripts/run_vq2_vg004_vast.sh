#!/usr/bin/env bash
# Source-locked Vast.ai launcher and epoch-resumable VG004 recurrent BC fit.
set -euo pipefail

VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?set VQ2_EXPECTED_COMMIT to the pushed source commit}"
VQ2_WORKSPACE_PATH="${VQ2_WORKSPACE_PATH:-/workspace/pufferlib-drone}"
VQ2_OUTPUT="${VQ2_OUTPUT:-$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg004_variable_gate_recurrent_bc_001}"
VQ2_STATE="$VQ2_OUTPUT/training_state.pt"
VQ2_DATASET="$VQ2_WORKSPACE_PATH/logs/drone_race_full_policy_six_gate_bootstrap/vq2_vg003_variable_gate_legal_bc_dataset_256"

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
    echo "VG004 requires the source-locked Vast virtual environment" >&2
    exit 2
fi
source .venv/bin/activate

# A published training state is the only resume authority. Do not install,
# test, or otherwise change the runtime after the initial epoch-zero snapshot.
if [ -f "$VQ2_STATE" ]; then
    python scripts/train_vq2_variable_gate_recurrent_bc.py \
        --output "$VQ2_OUTPUT" \
        --device cuda \
        --resume
    exit $?
fi

for command_name in git nvidia-smi python; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "missing required command: $command_name" >&2
        exit 2
    fi
done
if [ "$(nproc)" -lt 32 ]; then
    echo "VG004 requires at least 32 visible CPUs" >&2
    exit 2
fi
VQ2_RAM_GIB="$(awk '/MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)"
if [ "$VQ2_RAM_GIB" -lt 60 ]; then
    echo "VG004 requires approximately 64 GiB RAM" >&2
    exit 2
fi
VQ2_DISK_FREE_GIB="$(df -BG --output=avail "$VQ2_WORKSPACE_PATH" | tail -1 | tr -dc '0-9')"
if [ "$VQ2_DISK_FREE_GIB" -lt 20 ]; then
    echo "VG004 requires at least 20 GiB free" >&2
    exit 2
fi

test "$(sha256sum "$VQ2_DATASET/report.json" | cut -d" " -f1)" = \
    b8069d05cec63038659893a4020fa61ca1276ae035bc1a2ea5754f75e2298e85
test "$(sha256sum "$VQ2_DATASET/metadata.json" | cut -d" " -f1)" = \
    7a5d6359641c3e01f838e6fcdb57d32ddc18cc71ddbf1f8b3d0025309cabfe64

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the rented GPU")
name = torch.cuda.get_device_name(0)
if not any(candidate in name for candidate in ("RTX 4090", "A100", "RTX A6000")):
    raise SystemExit(f"unsupported VG004 GPU: {name}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={name}")
PY

python -m pytest -q \
    tests/test_vq2_recurrent_phase.py \
    tests/test_train_vq2_recurrent_bc.py \
    tests/test_train_vq2_variable_gate_recurrent_bc.py

python scripts/train_vq2_variable_gate_recurrent_bc.py \
    --output "$VQ2_OUTPUT" \
    --device cuda
