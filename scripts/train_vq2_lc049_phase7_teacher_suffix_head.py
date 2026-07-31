#!/usr/bin/env python3
"""Cheap phase-7 probe using LC040's reusable teacher suffix corpus."""

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


TAG = "vq2_lc049_phase7_teacher_suffix_head_001"
SCHEMA = "vq2_lc049_phase7_teacher_suffix_head_report_v1"
STATE_SCHEMA = "vq2_lc049_phase7_teacher_suffix_head_state_v1"
CHECKPOINT_SCHEMA = "vq2_lc049_phase7_teacher_suffix_head_checkpoint_v1"
TARGET_PHASES = (7,)
LC048 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc048_phase6_sequential_interpolation_bracket_001"
)
LC048_REPORT = LC048 / "report.json"
LC048_REPORT_SHA256 = (
    "0f5849a41f7554b88565caff758ec9a60e25d3c359d3c9603b45646c1dceb669"
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
    ROOT / "docs/vq2_lc049_phase7_teacher_suffix_head_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc049_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CONFIG = dataclasses.replace(base.CONFIG, seed=431490, chunk_rows=131_072)


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER,
        LC048_REPORT, PARENT_CHECKPOINT, PARENT_REPORT,
        DATASET_REPORT, FEATURES, LC041_REPORT,
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_lc040_phase4plus_teacher_features.py",
        ROOT / "scripts/train_vq2_lc013_index1_student_dagger_head.py",
        ROOT / "scripts/train_vq2_lc041_phase5plus_student_heads.py",
        ROOT / "scripts/eval_vq2_lc048_phase6_sequential_interpolation_bracket.py",
    )


def verify_inputs() -> dict[str, Any]:
    expected = {
        LC048_REPORT: LC048_REPORT_SHA256,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        LC041_REPORT: LC041_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC049 bound input changed: {path}")
    aggregate = json.loads(LC048_REPORT.read_text())
    parent = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    prior_fit = json.loads(LC041_REPORT.read_text())
    phases = dataset.get("feature_phase_records", [])
    if (
        aggregate.get("schema")
        != "vq2_lc048_phase6_sequential_interpolation_bracket_report_v1"
        or not aggregate.get("numerically_admitted")
        or aggregate.get("selected", {}).get("alpha") != 0.0025
        or aggregate.get("selected", {}).get("checkpoint_sha256")
        != PARENT_CHECKPOINT_SHA256
        or parent.get("schema") != "vq2_lc048_phase6_interpolation_screen_v1"
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema") != "vq2_lc040_phase4plus_teacher_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != FEATURE_RECORDS
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or FEATURES.stat().st_size
        != FEATURE_RECORDS * base.base.FEATURE_DTYPE.itemsize
        or len(phases) != LONG_COURSE_GATE_CAP + 1
        or phases[7] != 81_216
        or dataset.get("teacher_plant_actions_executed") != FEATURE_RECORDS
        or prior_fit.get("schema")
        != "vq2_lc041_phase5plus_student_heads_report_v1"
        or prior_fit.get("selected_validation", {}).get("phases", {}).get(
            "7", {}
        ).get("improvement_factor", 0.0) < 1.12
        or parent.get("maximum_raw_index") != 8
        or parent.get("crash_rate") != 0.03125
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or parent.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC040/LC041/LC048 do not authorize phase-7 probe")
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
    base.DATASET_REPORT_SCHEMA_EXPECTED = "vq2_lc040_phase4plus_teacher_report_v1"
    base.EXPECTED_FEATURE_RECORDS = FEATURE_RECORDS
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT, base.CONFIG = DEFAULT_OUTPUT, CONFIG
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), LC041_REPORT, LC048_REPORT)
    base.NEXT_AUTHORITY_ADMITTED = (
        "One reduced teacher-free paired phase-7 interpolation bracket from LC048."
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
