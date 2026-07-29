#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_ARCH="${NVCC_ARCH:-sm_86}"

TIMESTEPS="${TIMESTEPS:-10000000}"

if pgrep -af 'pufferlib.pufferl train' | grep -v "$$" >/dev/null; then
    echo "another PufferLib training process is already running" >&2
    exit 2
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    bash build.sh drone_race --float
fi

python -m pufferlib.pufferl train drone_race_vq1_v3391_telemetry \
    --train.total-timesteps "$TIMESTEPS" \
    "$@"
