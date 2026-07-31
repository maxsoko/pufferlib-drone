#!/usr/bin/env python3
"""Third phase-2 fit on LC028 states owned by the LC027 Puffer."""

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


TAG = "vq2_lc029_index2_student_dagger_head_001"
SCHEMA = "vq2_lc029_index2_student_dagger_head_report_v1"
STATE_SCHEMA = "vq2_lc029_index2_student_dagger_head_state_v1"
CHECKPOINT_SCHEMA = "vq2_lc029_index2_student_dagger_head_checkpoint_v1"
TARGET_PHASES = (2,)
LC027 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc027_phase2_interpolation_bracket_001"
)
PARENT_CHECKPOINT = LC027 / "a0p050/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "4cdd636ab0a1d6741614c5f2f74a545a213492306e90d5d89dffff0430240bb9"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "69845f35f66e1056f9228d6e0066272b3fe5f91a900c55af80713d5295e5da45"
)
DATASET = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc028_index2_student_dagger_features_001"
)
DATASET_REPORT = DATASET / "report.json"
DATASET_REPORT_SHA256 = (
    "ec8d760edeca4f07c4238f6edd76fb4db7ebc955bf0506f619020615b0ba1a1e"
)
FEATURES = DATASET / "features.bin"
FEATURES_SHA256 = (
    "4a74f8178a7f3abd33bba8f1ec2009245c53b5dcb06bf03b26267f96f7650701"
)
FEATURE_RECORDS = 431_450
PREREGISTRATION = (
    ROOT / "docs/vq2_lc029_index2_student_dagger_head_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc029_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CONFIG = dataclasses.replace(base.CONFIG, seed=431290)


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.CHECKPOINT_SCHEMA, base.TARGET_PHASES = CHECKPOINT_SCHEMA, TARGET_PHASES
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    base.PARENT_CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc027_phase2_interpolation_checkpoint_v1"
    )
    base.PARENT_REPORT_SCHEMA_EXPECTED = "vq2_lc027_phase2_interpolation_screen_v1"
    base.DATASET, base.DATASET_REPORT = DATASET, DATASET_REPORT
    base.DATASET_REPORT_SHA256 = DATASET_REPORT_SHA256
    base.DATASET_REPORT_SCHEMA_EXPECTED = "vq2_lc028_index2_student_dagger_report_v1"
    base.EXPECTED_FEATURE_RECORDS = FEATURE_RECORDS
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT, base.CONFIG = DEFAULT_OUTPUT, CONFIG
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        ROOT / "scripts/collect_vq2_lc028_index2_student_dagger_features.py",
    )
    base.NEXT_AUTHORITY_ADMITTED = (
        "One final paired phase-2 interpolation bracket; stop on no gain."
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
