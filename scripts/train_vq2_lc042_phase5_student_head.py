#!/usr/bin/env python3
"""Fit only phase 5 on the admitted LC040 continuation corpus."""

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

from pufferlib.vq2_public_phase import LONG_COURSE_GATE_CAP
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_lc013_index1_student_dagger_head as base


TAG = "vq2_lc042_phase5_student_head_001"
SCHEMA = "vq2_lc042_phase5_student_head_report_v1"
STATE_SCHEMA = "vq2_lc042_phase5_student_head_state_v1"
CHECKPOINT_SCHEMA = "vq2_lc042_phase5_student_head_checkpoint_v1"
TARGET_PHASES = (5,)
LC037 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc037_phase4_interpolation_bracket_001"
)
PARENT_CHECKPOINT = LC037 / "a0p005/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "da1940dd6bedee11e4eb6449bcd2a7844bb6ca3cdcf238900c1c6b22cd47507d"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "73e0bcd3fa320140aade4bfae1d0393544dd816d5e460bf6eeb6be107ff4f18f"
)
DATASET = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc040_phase4plus_teacher_features_001"
)
DATASET_REPORT = DATASET / "report.json"
DATASET_REPORT_SHA256 = (
    "7b814fab385873ae23507e19ecd4323375584c59740fd25c8316bb19aa68e06a"
)
FEATURES = DATASET / "features.bin"
FEATURES_SHA256 = (
    "50ad981fedfadf70fb35195c15369b30523c47a0e511923be31208df48b8922d"
)
FEATURE_RECORDS = 1_404_080
LC041_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc041_phase5plus_student_heads_001/report.json"
)
LC041_REPORT_SHA256 = (
    "b6c87b68a6f21c72bfcd52ec53d5848cb3f5b47d64280d51b40269c6e9aa3965"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc042_phase5_student_head_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc042_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CONFIG = dataclasses.replace(base.CONFIG, seed=431420, chunk_rows=131_072)


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER,
        PARENT_CHECKPOINT, PARENT_REPORT, DATASET_REPORT, FEATURES,
        LC041_REPORT,
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_lc040_phase4plus_teacher_features.py",
        ROOT / "scripts/train_vq2_lc013_index1_student_dagger_head.py",
        ROOT / "scripts/train_vq2_lc041_phase5plus_student_heads.py",
    )


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        LC041_REPORT: LC041_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC042 bound input changed: {path}")
    parent = json.loads(PARENT_REPORT.read_text())
    report = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC041_REPORT.read_text())
    phases = report.get("feature_phase_records", [])
    if (
        parent.get("schema") != "vq2_lc037_phase4_interpolation_screen_v1"
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or report.get("schema") != "vq2_lc040_phase4plus_teacher_report_v1"
        or not report.get("training_dataset_admitted")
        or report.get("feature_records") != FEATURE_RECORDS
        or report.get("feature_sha256") != FEATURES_SHA256
        or FEATURES.stat().st_size
        != FEATURE_RECORDS * base.base.FEATURE_DTYPE.itemsize
        or len(phases) != LONG_COURSE_GATE_CAP + 1
        or phases[5] != 80_144
        or report.get("teacher_plant_actions_executed") != FEATURE_RECORDS
        or rejected.get("schema")
        != "vq2_lc041_phase5plus_student_heads_report_v1"
        or rejected.get("numerically_admitted")
        or rejected.get("selected_validation", {}).get("phases", {}).get(
            "5", {}
        ).get("improvement_factor", 0.0) < 1.58
        or rejected.get("selected_validation", {}).get("phases", {}).get(
            "6", {}
        ).get("improvement_factor", 1.0) >= 1.0
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC040/LC041 do not authorize the isolated phase-5 fit")
    return report


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
        "vq2_lc037_phase4_interpolation_checkpoint_v1"
    )
    base.PARENT_REPORT_SCHEMA_EXPECTED = "vq2_lc037_phase4_interpolation_screen_v1"
    base.DATASET, base.DATASET_REPORT = DATASET, DATASET_REPORT
    base.DATASET_REPORT_SHA256 = DATASET_REPORT_SHA256
    base.DATASET_REPORT_SCHEMA_EXPECTED = "vq2_lc040_phase4plus_teacher_report_v1"
    base.EXPECTED_FEATURE_RECORDS = FEATURE_RECORDS
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT, base.CONFIG = DEFAULT_OUTPUT, CONFIG
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), LC041_REPORT)
    base.NEXT_AUTHORITY_ADMITTED = (
        "One teacher-free paired phase-5 interpolation bracket from LC037."
    )
    base.source_paths = source_paths
    base.verify_inputs = verify_inputs
    base.configure()


def fit(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
        resume: bool = False) -> dict[str, Any]:
    configure()
    return base.base.fit(
        output=output, device_name=device_name, resume=resume, config=CONFIG
    )


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
