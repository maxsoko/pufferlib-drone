#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pufferlib-drone
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1

.venv/bin/python scripts/eval_vq2_lc232_phase2_numpy_batch1_native.py --target 3
if .venv/bin/python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path(
    "logs/drone_race_full_policy_six_gate_bootstrap/"
    "vq2_lc232_phase2_numpy_batch1_native_raw3_001/report.json"
).read_text())
raise SystemExit(0 if report.get("milestone_admitted") else 1)
PY
then
    .venv/bin/python scripts/eval_vq2_lc232_phase2_numpy_batch1_native.py --target 24
fi
