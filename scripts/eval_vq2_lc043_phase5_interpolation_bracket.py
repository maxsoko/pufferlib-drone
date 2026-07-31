#!/usr/bin/env python3
"""Teacher-free paired bracket toward the LC042 fitted phase-5 head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.eval_vq2_lc024_phase2_interpolation_bracket as base


TAG = "vq2_lc043_phase5_interpolation_bracket_001"
SCHEMA = "vq2_lc043_phase5_interpolation_bracket_report_v1"
CHILD_SCHEMA = "vq2_lc043_phase5_interpolation_screen_v1"
CHECKPOINT_SCHEMA = "vq2_lc043_phase5_interpolation_checkpoint_v1"
ALPHAS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25)
SEED = 431430
TARGET_PHASE = 5
LC037 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc037_phase4_interpolation_bracket_001"
)
LC037_REPORT = LC037 / "report.json"
LC037_REPORT_SHA256 = (
    "5678e4550d2301e8befb1360d4c2b6b0a1b5bec7e984c8f09dd9aae00f6c114b"
)
PARENT_CHECKPOINT = LC037 / "a0p005/policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = (
    "da1940dd6bedee11e4eb6449bcd2a7844bb6ca3cdcf238900c1c6b22cd47507d"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "73e0bcd3fa320140aade4bfae1d0393544dd816d5e460bf6eeb6be107ff4f18f"
)
LC040_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc040_phase4plus_teacher_features_001/report.json"
)
LC040_REPORT_SHA256 = (
    "7b814fab385873ae23507e19ecd4323375584c59740fd25c8316bb19aa68e06a"
)
FIT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc042_phase5_student_head_001/policy_best.pt"
)
FIT_CHECKPOINT_SHA256 = (
    "189a7f618a9076fb1acd3613376255a5111162a5d702d069f92f1db3192d343e"
)
FIT_REPORT = FIT_CHECKPOINT.parent / "report.json"
FIT_REPORT_SHA256 = (
    "18ba235c154ff769981a58061e43f4b53b4a58da59b028ff15dae0f45eaa1803"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc043_phase5_interpolation_bracket_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc043_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


def verify_inputs() -> None:
    expected = {
        LC037_REPORT: LC037_REPORT_SHA256,
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC040_REPORT: LC040_REPORT_SHA256,
        FIT_CHECKPOINT: FIT_CHECKPOINT_SHA256,
        FIT_REPORT: FIT_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC043 bound input changed: {path}")
    aggregate = json.loads(LC037_REPORT.read_text())
    dataset = json.loads(LC040_REPORT.read_text())
    fit = json.loads(FIT_REPORT.read_text())
    if (
        aggregate.get("schema")
        != "vq2_lc037_phase4_interpolation_bracket_report_v1"
        or not aggregate.get("numerically_admitted")
        or aggregate.get("selected", {}).get("alpha") != 0.005
        or aggregate.get("selected", {}).get("checkpoint_sha256")
        != PARENT_CHECKPOINT_SHA256
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_phase_records", [])[TARGET_PHASE] != 80_144
        or dataset.get("teacher_plant_actions_executed") != 1_404_080
        or dataset.get("safety", {}).get("runtime_teacher_authorized")
        or fit.get("schema") != "vq2_lc042_phase5_student_head_report_v1"
        or not fit.get("numerically_admitted")
        or fit.get("checkpoint_sha256") != FIT_CHECKPOINT_SHA256
        or fit.get("selected_validation", {}).get("phases", {}).get(
            "5", {}
        ).get("improvement_factor", 0.0) < 1.55
        or fit.get("safety", {}).get("flight_sim_packets_sent") != 0
        or fit.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC037/LC040/LC042 do not authorize LC043")


def configure() -> None:
    base.TAG, base.SCHEMA, base.CHILD_SCHEMA = TAG, SCHEMA, CHILD_SCHEMA
    base.CHECKPOINT_SCHEMA, base.ALPHAS, base.SEED = (
        CHECKPOINT_SCHEMA, ALPHAS, SEED,
    )
    base.TARGET_PHASE = TARGET_PHASE
    base.LC021_REPORT, base.LC021_REPORT_SHA256 = (
        LC037_REPORT, LC037_REPORT_SHA256,
    )
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    base.LC022_REPORT, base.LC022_REPORT_SHA256 = (
        LC040_REPORT, LC040_REPORT_SHA256,
    )
    base.FIT_CHECKPOINT, base.FIT_CHECKPOINT_SHA256 = (
        FIT_CHECKPOINT, FIT_CHECKPOINT_SHA256,
    )
    base.FIT_REPORT, base.FIT_REPORT_SHA256 = FIT_REPORT, FIT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.PARENT_REPORT_SCHEMA_EXPECTED = (
        "vq2_lc037_phase4_interpolation_bracket_report_v1"
    )
    base.PARENT_SELECTION_FIELD, base.PARENT_SELECTION_VALUE = "alpha", 0.005
    base.PARENT_CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc037_phase4_interpolation_checkpoint_v1"
    )
    base.DATASET_EXPECTED_RECORDS = 80_144
    base.FIT_REPORT_SCHEMA_EXPECTED = "vq2_lc042_phase5_student_head_report_v1"
    base.FIT_CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc042_phase5_student_head_checkpoint_v1"
    )
    base.FIT_MINIMUM_IMPROVEMENT = 1.55
    base.HISTORICAL_MINIMUM_MEAN_GATES = 3.71875
    base.HISTORICAL_MINIMUM_MAX_INDEX = 7
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(),)
    base.verify_inputs = verify_inputs


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
