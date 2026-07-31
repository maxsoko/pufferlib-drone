#!/usr/bin/env python3
"""Narrow paired bracket toward the second phase-2 student-state fit."""

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


TAG = "vq2_lc027_phase2_interpolation_bracket_001"
SCHEMA = "vq2_lc027_phase2_interpolation_bracket_report_v1"
CHILD_SCHEMA = "vq2_lc027_phase2_interpolation_screen_v1"
CHECKPOINT_SCHEMA = "vq2_lc027_phase2_interpolation_checkpoint_v1"
ALPHAS = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.05, 0.10)
SEED = 431270
LC024 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc024_phase2_interpolation_bracket_001"
)
LC024_REPORT = LC024 / "report.json"
LC024_REPORT_SHA256 = (
    "c2b1fcfcd8a5fe552bf90461493ecfb1f71c3a7de6853475961a430951a41fd9"
)
PARENT_CHECKPOINT = LC024 / "a0p010/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "167e58f16b193ab1d2e64211a05482bfbdb6bd005d2fad99c6284fe517fae786"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "b133d388d2ddeeff809e102f13900b7acfeddc294a56393d2d0ce17b4353edcf"
)
LC025_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc025_index2_student_dagger_features_001/report.json"
)
LC025_REPORT_SHA256 = (
    "a505af17e26803a794491f91051d1797e40ad4ba2d8e1e2622cec449ab4096c4"
)
FIT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc026_index2_student_dagger_head_001/policy_best.pt"
)
FIT_CHECKPOINT_SHA256 = (
    "634b1ca43990d1133aa66345dd9ae35baf86d09f379c5dcbe514214c3fb41147"
)
FIT_REPORT = FIT_CHECKPOINT.parent / "report.json"
FIT_REPORT_SHA256 = (
    "ee256f5cf4d4cae482f3cf0b9607d55774c42e395e688997dca3e73d23ce30e7"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc027_phase2_interpolation_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc027_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def configure() -> None:
    base.TAG = TAG
    base.SCHEMA = SCHEMA
    base.CHILD_SCHEMA = CHILD_SCHEMA
    base.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    base.ALPHAS = ALPHAS
    base.SEED = SEED
    base.LC021_REPORT = LC024_REPORT
    base.LC021_REPORT_SHA256 = LC024_REPORT_SHA256
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT = PARENT_REPORT
    base.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.LC022_REPORT = LC025_REPORT
    base.LC022_REPORT_SHA256 = LC025_REPORT_SHA256
    base.FIT_CHECKPOINT = FIT_CHECKPOINT
    base.FIT_CHECKPOINT_SHA256 = FIT_CHECKPOINT_SHA256
    base.FIT_REPORT = FIT_REPORT
    base.FIT_REPORT_SHA256 = FIT_REPORT_SHA256
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.PARENT_REPORT_SCHEMA_EXPECTED = (
        "vq2_lc024_phase2_interpolation_bracket_report_v1"
    )
    base.PARENT_SELECTION_FIELD = "alpha"
    base.PARENT_SELECTION_VALUE = 0.01
    base.PARENT_CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc024_phase2_interpolation_checkpoint_v1"
    )
    base.DATASET_EXPECTED_RECORDS = 528_846
    base.FIT_REPORT_SCHEMA_EXPECTED = (
        "vq2_lc026_index2_student_dagger_head_report_v1"
    )
    base.FIT_CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc026_index2_student_dagger_head_checkpoint_v1"
    )
    base.FIT_MINIMUM_IMPROVEMENT = 2.27
    base.HISTORICAL_MINIMUM_MEAN_GATES = 3.25
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
