#!/usr/bin/env python3
"""LC040 corrected phase-4 teacher boundary for the long-course corpus."""

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
import scripts.collect_vq2_lc038_phase4plus_teacher_features as base


TAG = "vq2_lc040_phase4plus_teacher_features_001"
SCHEMA = "vq2_lc040_phase4plus_teacher_report_v1"
STATE_SCHEMA = "vq2_lc040_phase4plus_teacher_state_v1"
SEED = 431400
LC039_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc039_phase4plus_teacher_features_001/report.json"
)
LC039_REPORT_SHA256 = (
    "23f11c5a10dd6ef35229bbc90b4204d80eaf0f014126c2268994852260e2c6fe"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc040_phase4plus_teacher_features_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc040_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
BASE_VERIFY_INPUTS = base.verify_inputs


def verify_inputs() -> None:
    BASE_VERIFY_INPUTS()
    if sha256_path(LC039_REPORT) != LC039_REPORT_SHA256:
        raise RuntimeError("LC040 bound LC039 diagnosis changed")
    failed = json.loads(LC039_REPORT.read_text())
    if (
        failed.get("schema") != "vq2_lc039_phase4plus_teacher_report_v1"
        or failed.get("training_dataset_admitted")
        or failed.get("failed_admission_predicates")
        != ["records_match_teacher_plant_actions"]
        or failed.get("feature_records") != 3_166_365
        or failed.get("teacher_query_actions_recorded") != 3_166_365
        or failed.get("teacher_plant_actions_executed") != 3_631_517
        or failed.get("minimum_teacher_phase_index") != 4
        or failed.get("maximum_teacher_phase_index") != 23
        or failed.get("safety", {}).get("flight_sim_packets_sent") != 0
        or failed.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC039 does not establish the phase-boundary bug")


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.SEED = SEED
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), LC039_REPORT)
    base.verify_inputs = verify_inputs
    base.configure()


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
            resume: bool = False) -> dict[str, Any]:
    configure()
    return base.base.collect(
        output=output, device_name=device_name, resume=resume
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_dataset_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
