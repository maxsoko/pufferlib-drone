#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTEPS="${TIMESTEPS:-65536}"
ENVS="${ENVS:-drone drone_race}"
PRECHECK_ONLY=0

for arg in "$@"; do
    case "$arg" in
        --precheck-only) PRECHECK_ONLY=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$ROOT"

if [ "$(uname -s)" != "Linux" ]; then
    echo "native GPU validation requires Linux; current platform is $(uname -s)" >&2
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "native GPU validation requires nvcc on PATH" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "native GPU validation requires nvidia-smi on PATH" >&2
    exit 1
fi

python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "pybind11", "numpy") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing Python packages for native GPU validation: {', '.join(missing)}")
PY

nvidia-smi
nvcc --version

if [ "$PRECHECK_ONLY" = "1" ]; then
    echo "native GPU precheck passed"
    exit 0
fi

for env in $ENVS; do
    echo "==> building native CUDA backend for $env"
    bash build.sh "$env"

    echo "==> native training smoke for $env (${TIMESTEPS} timesteps)"
    python -m pufferlib.pufferl train "$env" \
        --train.total-timesteps "$TIMESTEPS"
done

echo "native GPU validation complete"
