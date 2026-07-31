#!/usr/bin/env python3
"""Paired closed-loop bracket toward the LC032 fitted phase-4 head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_vq2_lc024_phase2_interpolation_bracket as base


TAG = "vq2_lc033_phase4_interpolation_bracket_001"
SCHEMA = "vq2_lc033_phase4_interpolation_bracket_report_v1"
CHILD_SCHEMA = "vq2_lc033_phase4_interpolation_screen_v1"
CHECKPOINT_SCHEMA = "vq2_lc033_phase4_interpolation_checkpoint_v1"
ALPHAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.0)
SEED = 431330
TARGET_PHASE = 4
LC027 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc027_phase2_interpolation_bracket_001"
)
LC027_REPORT = LC027 / "report.json"
LC027_REPORT_SHA256 = (
    "e161ca51447e4a324529f37b2d8175fd6d6667e77f67f3b61f805452100bbef2"
)
PARENT_CHECKPOINT = LC027 / "a0p050/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "4cdd636ab0a1d6741614c5f2f74a545a213492306e90d5d89dffff0430240bb9"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "69845f35f66e1056f9228d6e0066272b3fe5f91a900c55af80713d5295e5da45"
)
LC031_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc031_index4_student_dagger_features_001/report.json"
)
LC031_REPORT_SHA256 = (
    "273f1096b8726aaf2f10ccdb240edbd37ef72a187071e84f897b122704e2cc0d"
)
FIT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc032_index4_student_dagger_head_001/policy_best.pt"
)
FIT_CHECKPOINT_SHA256 = (
    "85d10385cb05d7bedc6ee01dd2985606da170a79ca73cab4df1bf622e3ced9bc"
)
FIT_REPORT = FIT_CHECKPOINT.parent / "report.json"
FIT_REPORT_SHA256 = (
    "c17d9b0179d924643800d0062c13813055ba91af3d5d7db6035d804e5a97f262"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc033_phase4_interpolation_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc033_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def configure() -> None:
    base.TAG, base.SCHEMA, base.CHILD_SCHEMA = TAG, SCHEMA, CHILD_SCHEMA
    base.CHECKPOINT_SCHEMA, base.ALPHAS, base.SEED = (
        CHECKPOINT_SCHEMA, ALPHAS, SEED,
    )
    base.TARGET_PHASE = TARGET_PHASE
    base.LC021_REPORT, base.LC021_REPORT_SHA256 = (
        LC027_REPORT, LC027_REPORT_SHA256,
    )
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    base.LC022_REPORT, base.LC022_REPORT_SHA256 = (
        LC031_REPORT, LC031_REPORT_SHA256,
    )
    base.FIT_CHECKPOINT, base.FIT_CHECKPOINT_SHA256 = (
        FIT_CHECKPOINT, FIT_CHECKPOINT_SHA256,
    )
    base.FIT_REPORT, base.FIT_REPORT_SHA256 = FIT_REPORT, FIT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.PARENT_REPORT_SCHEMA_EXPECTED = (
        "vq2_lc027_phase2_interpolation_bracket_report_v1"
    )
    base.PARENT_SELECTION_FIELD, base.PARENT_SELECTION_VALUE = "alpha", 0.05
    base.PARENT_CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc027_phase2_interpolation_checkpoint_v1"
    )
    base.DATASET_EXPECTED_RECORDS = 79_056
    base.FIT_REPORT_SCHEMA_EXPECTED = (
        "vq2_lc032_index4_student_dagger_head_report_v1"
    )
    base.FIT_CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc032_index4_student_dagger_head_checkpoint_v1"
    )
    base.FIT_MINIMUM_IMPROVEMENT = 1.16
    base.HISTORICAL_MINIMUM_MEAN_GATES = 3.5
    base.HISTORICAL_MINIMUM_MAX_INDEX = 6
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)


def run(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    configure()
    return base.run(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
