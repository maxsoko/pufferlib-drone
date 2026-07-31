#!/usr/bin/env python3
"""LC039 retry of the long-course corpus with the 32-phase counter bound."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_vq2_lc038_phase4plus_teacher_features as base


TAG = "vq2_lc039_phase4plus_teacher_features_001"
SCHEMA = "vq2_lc039_phase4plus_teacher_report_v1"
STATE_SCHEMA = "vq2_lc039_phase4plus_teacher_state_v1"
SEED = 431390
PREREGISTRATION = (
    ROOT / "docs/vq2_lc039_phase4plus_teacher_features_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc039_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.SEED = SEED
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)
    base.configure()


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
            resume: bool = False) -> dict[str, Any]:
    configure()
    return base.base.collect(
        output=output, device_name=device_name, resume=resume
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_dataset_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
