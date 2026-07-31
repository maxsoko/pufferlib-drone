#!/usr/bin/env python3
"""Fit phase 7 with equal weight for each LC052 student trajectory."""

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


TAG = "vq2_lc054_phase7_equal_trajectory_head_001"
SCHEMA = "vq2_lc054_phase7_equal_trajectory_head_report_v1"
STATE_SCHEMA = "vq2_lc054_phase7_equal_trajectory_head_state_v1"
CHECKPOINT_SCHEMA = "vq2_lc054_phase7_equal_trajectory_head_checkpoint_v1"
TARGET_PHASES = (7,)
LC048 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc048_phase6_sequential_interpolation_bracket_001"
)
PARENT_CHECKPOINT = LC048 / "a0p003/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "5a2046a856aa29e8c8bff81789664ff68cd44d67e5ad6e6bf90ba02cf9dec084"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "b2fb203ca9a49e74371694da71f8bd72757a1d0aab3d930ce5c2c8afdd85b3de"
)
DATASET = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc052_index7_capped_student_features_001"
)
DATASET_REPORT = DATASET / "report.json"
DATASET_REPORT_SHA256 = (
    "66e81b3df80ca6e8fe99551c0781b92805187d1f741feb6ff268f415cdcace2d"
)
FEATURES = DATASET / "features.bin"
FEATURES_SHA256 = (
    "50a16280a80bea34a9f70008b10bfdddc5b90bc62e5871386a45f2d066dbbb4c"
)
FEATURE_RECORDS = 10_000
LC053_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc053_phase7_student_head_001/report.json"
)
LC053_REPORT_SHA256 = (
    "f61917e4565dc4bb4bd13584cc869886a15640051545fe050898097d59a93df0"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc054_phase7_equal_trajectory_head_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc054_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CONFIG = dataclasses.replace(
    base.CONFIG,
    seed=431540,
    epochs=10,
    learning_rate=1e-4,
    chunk_rows=FEATURE_RECORDS,
    validation_agent_group_size=1,
    validation_agent_modulus=8,
    validation_agent_remainder=3,
)


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER,
        PARENT_CHECKPOINT, PARENT_REPORT, DATASET_REPORT, FEATURES,
        LC053_REPORT,
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_lc052_index7_capped_student_features.py",
        ROOT / "scripts/train_vq2_lc013_index1_student_dagger_head.py",
        ROOT / "scripts/train_vq2_vg068_indexed_mlp_intervention_features.py",
    )


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        LC053_REPORT: LC053_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC054 bound input changed: {path}")
    parent = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC053_REPORT.read_text())
    phases = dataset.get("feature_phase_records", [])
    if (
        parent.get("schema") != "vq2_lc048_phase6_interpolation_screen_v1"
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema")
        != "vq2_lc052_index7_capped_student_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("failed_admission_predicates")
        or dataset.get("feature_records") != FEATURE_RECORDS
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or FEATURES.stat().st_size
        != FEATURE_RECORDS * base.base.FEATURE_DTYPE.itemsize
        or len(phases) != LONG_COURSE_GATE_CAP + 1
        or phases[7] != FEATURE_RECORDS
        or sum(value for index, value in enumerate(phases) if index != 7) != 0
        or dataset.get("query_agents") != 6
        or dataset.get("teacher_plant_actions_executed") != 0
        or dataset.get("student_plant_actions_executed")
        != dataset.get("total_plant_actions_executed")
        or rejected.get("schema")
        != "vq2_lc053_phase7_student_head_report_v1"
        or rejected.get("numerically_admitted")
        or rejected.get("selected_validation", {}).get("phases", {}).get(
            "7", {}
        ).get("improvement_factor") != 0.9585109679776481
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC048/LC052/LC053 do not authorize LC054")
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
        "vq2_lc048_phase6_interpolation_checkpoint_v1"
    )
    base.PARENT_REPORT_SCHEMA_EXPECTED = "vq2_lc048_phase6_interpolation_screen_v1"
    base.DATASET, base.DATASET_REPORT = DATASET, DATASET_REPORT
    base.DATASET_REPORT_SHA256 = DATASET_REPORT_SHA256
    base.DATASET_REPORT_SCHEMA_EXPECTED = "vq2_lc052_index7_capped_student_report_v1"
    base.EXPECTED_FEATURE_RECORDS = FEATURE_RECORDS
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT, base.CONFIG = DEFAULT_OUTPUT, CONFIG
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), LC053_REPORT)
    base.NEXT_AUTHORITY_ADMITTED = (
        "One reduced teacher-free paired phase-7 interpolation bracket from LC048."
    )
    base.source_paths = source_paths
    base.verify_inputs = verify_inputs
    base.configure()
    base.base.EQUAL_AGENT_WEIGHTING = True


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
