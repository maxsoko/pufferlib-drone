#!/usr/bin/env python3
"""Fit only the LC021 public-index-2 head on LC022 student-state labels."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_vq2_lc013_index1_student_dagger_head as base


TAG = "vq2_lc023_index2_student_dagger_head_001"
SCHEMA = "vq2_lc023_index2_student_dagger_head_report_v1"
STATE_SCHEMA = "vq2_lc023_index2_student_dagger_head_state_v1"
CHECKPOINT_SCHEMA = "vq2_lc023_index2_student_dagger_head_checkpoint_v1"
TARGET_PHASES = (2,)
LC021 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc021_cumulative_head_rollback_ladder_001"
)
PARENT_CHECKPOINT = LC021 / "through_05/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "a32cc7705145c47b63c5126128eaa1296a52bfb175b8bb7868b9839041275ca4"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "bd45e3ec075c4574fbc136a00ddb2b57830a2b96ef781d11e46959b8d7ab53fa"
)
PARENT_CHECKPOINT_SCHEMA = "vq2_lc021_cumulative_head_rollback_checkpoint_v1"
PARENT_REPORT_SCHEMA = "vq2_lc021_cumulative_head_rollback_screen_v1"
DATASET = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc022_index2_student_dagger_features_001"
)
DATASET_REPORT = DATASET / "report.json"
DATASET_REPORT_SHA256 = (
    "833d4eed8e4dc68b48d1efd86d1551e6296d2ffc2225fdccef0e54edab5515b3"
)
DATASET_REPORT_SCHEMA = "vq2_lc022_index2_student_dagger_report_v1"
FEATURES = DATASET / "features.bin"
FEATURES_SHA256 = (
    "2fbfdc5555849772ffb4fd54393132c26076e12c02746c09e9711ecbdd6aeeb8"
)
FEATURE_RECORDS = 649_205
PREREGISTRATION = (
    ROOT / "docs/vq2_lc023_index2_student_dagger_head_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc023_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CONFIG = dataclasses.replace(base.CONFIG, seed=431230)


def configure() -> None:
    base.TAG = TAG
    base.SCHEMA = SCHEMA
    base.STATE_SCHEMA = STATE_SCHEMA
    base.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    base.TARGET_PHASES = TARGET_PHASES
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT = PARENT_REPORT
    base.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.PARENT_CHECKPOINT_SCHEMA_EXPECTED = PARENT_CHECKPOINT_SCHEMA
    base.PARENT_REPORT_SCHEMA_EXPECTED = PARENT_REPORT_SCHEMA
    base.DATASET = DATASET
    base.DATASET_REPORT = DATASET_REPORT
    base.DATASET_REPORT_SHA256 = DATASET_REPORT_SHA256
    base.DATASET_REPORT_SCHEMA_EXPECTED = DATASET_REPORT_SCHEMA
    base.EXPECTED_FEATURE_RECORDS = FEATURE_RECORDS
    base.FEATURES = FEATURES
    base.FEATURES_SHA256 = FEATURES_SHA256
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.CONFIG = CONFIG
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        ROOT / "scripts/collect_vq2_lc022_index2_student_dagger_features.py",
    )
    base.NEXT_AUTHORITY_ADMITTED = (
        "One paired bounded interpolation bracket toward the fitted phase-2 head."
    )


def fit(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    configure()
    return base.fit(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = fit(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
