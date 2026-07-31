#!/usr/bin/env python3
"""Fit only phase 4 on LC035 states owned by the LC033 Puffer."""

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


TAG = "vq2_lc036_index4_student_dagger_head_001"
SCHEMA = "vq2_lc036_index4_student_dagger_head_report_v1"
STATE_SCHEMA = "vq2_lc036_index4_student_dagger_head_state_v1"
CHECKPOINT_SCHEMA = "vq2_lc036_index4_student_dagger_head_checkpoint_v1"
TARGET_PHASES = (4,)
LC033 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc033_phase4_interpolation_bracket_001"
)
PARENT_CHECKPOINT = LC033 / "a0p250/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "5b344710e28edb9e2bc080efc4101b398c554cf509a6cdc436b0c1baded53678"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "63839834e56c59d480a2c0ac13991515aa21d7f533b9793cd3cec16e4b1ab16b"
)
DATASET = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc035_index4_student_dagger_features_001"
)
DATASET_REPORT = DATASET / "report.json"
DATASET_REPORT_SHA256 = (
    "ffef510d2f824e54b8d32c85eee356484fe5c1e21b492df90717fd192b4e1199"
)
FEATURES = DATASET / "features.bin"
FEATURES_SHA256 = (
    "f1fca8e84898fcc6102bf8cad9a159a6c82060fc033a46e3d456edf90aba0620"
)
FEATURE_RECORDS = 77_211
PREREGISTRATION = (
    ROOT / "docs/vq2_lc036_index4_student_dagger_head_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc036_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CONFIG = dataclasses.replace(base.CONFIG, seed=431360)


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
        "vq2_lc033_phase4_interpolation_checkpoint_v1"
    )
    base.PARENT_REPORT_SCHEMA_EXPECTED = "vq2_lc033_phase4_interpolation_screen_v1"
    base.DATASET, base.DATASET_REPORT = DATASET, DATASET_REPORT
    base.DATASET_REPORT_SHA256 = DATASET_REPORT_SHA256
    base.DATASET_REPORT_SCHEMA_EXPECTED = "vq2_lc035_index4_student_dagger_report_v1"
    base.EXPECTED_FEATURE_RECORDS = FEATURE_RECORDS
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT, base.CONFIG = DEFAULT_OUTPUT, CONFIG
    base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(),
        ROOT / "scripts/collect_vq2_lc035_index4_student_dagger_features.py",
    )
    base.NEXT_AUTHORITY_ADMITTED = (
        "One paired phase-4 interpolation bracket from the LC033 parent."
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
