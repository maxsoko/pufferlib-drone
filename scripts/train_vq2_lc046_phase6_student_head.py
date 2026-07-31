#!/usr/bin/env python3
"""Fit phase 6 on LC045's Puffer-owned full-start states."""

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


TAG = "vq2_lc046_phase6_student_head_001"
SCHEMA = "vq2_lc046_phase6_student_head_report_v1"
STATE_SCHEMA = "vq2_lc046_phase6_student_head_state_v1"
CHECKPOINT_SCHEMA = "vq2_lc046_phase6_student_head_checkpoint_v1"
TARGET_PHASES = (6,)
LC043 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc043_phase5_interpolation_bracket_001"
)
PARENT_CHECKPOINT = LC043 / "a0p005/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "d2f9c235cc0a966d8d96ddca7513026e6689cf14aeb8c0cff22891769b93ad52"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "35f14abb43d7e6a9db7cac4370102131dfcc6c94bde68bc83ccd123e6774366d"
)
DATASET = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc045_index6_student_dagger_features_001"
)
DATASET_REPORT = DATASET / "report.json"
DATASET_REPORT_SHA256 = (
    "2abc1bf32d5e8812779fcfe4ed495f2e947d6098e99b07ed15332bfbbcfaebfa"
)
FEATURES = DATASET / "features.bin"
FEATURES_SHA256 = (
    "63643202ea60201dd007d95b1d4c345a96be51abb8d5f62e4ecfdbcab4314593"
)
FEATURE_RECORDS = 189_009
LC044_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc044_phase6_student_head_001/report.json"
)
LC044_REPORT_SHA256 = (
    "d7592fa112e0d3bc4d8cdef5281ac80f1391535bf9540001cc9608d22dd16f60"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc046_phase6_student_head_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc046_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CONFIG = dataclasses.replace(base.CONFIG, seed=431460, chunk_rows=32_768)


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER,
        PARENT_CHECKPOINT, PARENT_REPORT, DATASET_REPORT, FEATURES,
        LC044_REPORT,
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_lc045_index6_student_dagger_features.py",
        ROOT / "scripts/train_vq2_lc013_index1_student_dagger_head.py",
        ROOT / "scripts/train_vq2_lc044_phase6_student_head.py",
    )


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        LC044_REPORT: LC044_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC046 bound input changed: {path}")
    parent = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC044_REPORT.read_text())
    phases = dataset.get("feature_phase_records", [])
    if (
        parent.get("schema") != "vq2_lc043_phase5_interpolation_screen_v1"
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or parent.get("maximum_raw_index") != 8
        or parent.get("crash_rate") != 0.03125
        or dataset.get("schema")
        != "vq2_lc045_index6_student_dagger_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("failed_admission_predicates")
        or dataset.get("feature_records") != FEATURE_RECORDS
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or FEATURES.stat().st_size
        != FEATURE_RECORDS * base.base.FEATURE_DTYPE.itemsize
        or len(phases) != LONG_COURSE_GATE_CAP + 1
        or phases[6] != FEATURE_RECORDS
        or sum(value for index, value in enumerate(phases) if index != 6) != 0
        or dataset.get("teacher_plant_actions_executed") != 0
        or dataset.get("student_plant_actions_executed")
        != dataset.get("total_plant_actions_executed")
        or dataset.get("metrics", {}).get("env/ordered_gate5_sampled", 0.0)
        < 0.10
        or rejected.get("schema") != "vq2_lc044_phase6_student_head_report_v1"
        or rejected.get("numerically_admitted")
        or rejected.get("selected_validation", {}).get("phases", {}).get(
            "6", {}
        ).get("improvement_factor") != 1.0173365731903161
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC043--LC045 do not authorize the phase-6 fit")
    return dataset


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
        "vq2_lc043_phase5_interpolation_checkpoint_v1"
    )
    base.PARENT_REPORT_SCHEMA_EXPECTED = "vq2_lc043_phase5_interpolation_screen_v1"
    base.DATASET, base.DATASET_REPORT = DATASET, DATASET_REPORT
    base.DATASET_REPORT_SHA256 = DATASET_REPORT_SHA256
    base.DATASET_REPORT_SCHEMA_EXPECTED = (
        "vq2_lc045_index6_student_dagger_report_v1"
    )
    base.EXPECTED_FEATURE_RECORDS = FEATURE_RECORDS
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT, base.CONFIG = DEFAULT_OUTPUT, CONFIG
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), LC044_REPORT)
    base.NEXT_AUTHORITY_ADMITTED = (
        "One teacher-free paired phase-6 interpolation bracket from LC043."
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
