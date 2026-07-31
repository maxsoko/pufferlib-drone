#!/usr/bin/env python3
"""Paired bracket toward the LC036 second fitted phase-4 head."""

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


TAG = "vq2_lc037_phase4_interpolation_bracket_001"
SCHEMA = "vq2_lc037_phase4_interpolation_bracket_report_v1"
CHILD_SCHEMA = "vq2_lc037_phase4_interpolation_screen_v1"
CHECKPOINT_SCHEMA = "vq2_lc037_phase4_interpolation_checkpoint_v1"
ALPHAS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25)
SEED = 431370
TARGET_PHASE = 4
LC033 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc033_phase4_interpolation_bracket_001"
)
LC033_REPORT = LC033 / "report.json"
LC033_REPORT_SHA256 = (
    "57d358bd8acb0739853d95f2f4cd5872ca1db7d0e4415cfdcc19a83ef1941de2"
)
PARENT_CHECKPOINT = LC033 / "a0p250/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "5b344710e28edb9e2bc080efc4101b398c554cf509a6cdc436b0c1baded53678"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "63839834e56c59d480a2c0ac13991515aa21d7f533b9793cd3cec16e4b1ab16b"
)
LC035_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc035_index4_student_dagger_features_001/report.json"
)
LC035_REPORT_SHA256 = (
    "ffef510d2f824e54b8d32c85eee356484fe5c1e21b492df90717fd192b4e1199"
)
FIT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc036_index4_student_dagger_head_001/policy_best.pt"
)
FIT_CHECKPOINT_SHA256 = (
    "a7c5c36d8d16e0fa7d311e966fd598c6f6c05a52ef2b65836f1d004225e05e56"
)
FIT_REPORT = FIT_CHECKPOINT.parent / "report.json"
FIT_REPORT_SHA256 = (
    "2e0ab0bb6049df9d218ff113fd2ba20440dc8dfb695a9cb8a1a1ac7f5bbe5dc6"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc037_phase4_interpolation_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc037_vast.sh"
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
        LC033_REPORT, LC033_REPORT_SHA256,
    )
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    base.LC022_REPORT, base.LC022_REPORT_SHA256 = (
        LC035_REPORT, LC035_REPORT_SHA256,
    )
    base.FIT_CHECKPOINT, base.FIT_CHECKPOINT_SHA256 = (
        FIT_CHECKPOINT, FIT_CHECKPOINT_SHA256,
    )
    base.FIT_REPORT, base.FIT_REPORT_SHA256 = FIT_REPORT, FIT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.PARENT_REPORT_SCHEMA_EXPECTED = (
        "vq2_lc033_phase4_interpolation_bracket_report_v1"
    )
    base.PARENT_SELECTION_FIELD, base.PARENT_SELECTION_VALUE = "alpha", 0.25
    base.PARENT_CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc033_phase4_interpolation_checkpoint_v1"
    )
    base.DATASET_EXPECTED_RECORDS = 77_211
    base.FIT_REPORT_SCHEMA_EXPECTED = (
        "vq2_lc036_index4_student_dagger_head_report_v1"
    )
    base.FIT_CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc036_index4_student_dagger_head_checkpoint_v1"
    )
    base.FIT_MINIMUM_IMPROVEMENT = 1.19
    base.HISTORICAL_MINIMUM_MEAN_GATES = 3.59375
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
