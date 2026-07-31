#!/usr/bin/env python3
"""Run the 360-second horizon-corrected warmed teacher intervention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_vg062_warmed_teacher_intervention_features as core


BASE_VERIFY_INPUTS = core.verify_inputs
TAG = "vq2_vg063_horizon_corrected_intervention_features_001"
REPORT_SCHEMA = "vq2_vg063_horizon_corrected_intervention_report_v1"
STATE_SCHEMA = "vq2_vg063_horizon_corrected_intervention_state_v1"
SEED = 429198
STEP_LIMIT = 23040
MINIMUM_SUCCESS_RATE = 0.99
MINIMUM_RECORDS = 1_000_000
REJECTION = ROOT / "docs/vq2_vg062_warmed_teacher_intervention_rejection_2026-07-31.json"
REJECTION_SHA256 = "397b614af2bf3aac3d8ef6e9a1dd2180a08af865d17b53d0685d4d309916ce3f"
PREREGISTRATION = ROOT / "docs/vq2_vg063_horizon_corrected_intervention_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg063_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_vg063_horizon_corrected_intervention_features.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    BASE_VERIFY_INPUTS()
    if sha256_path(REJECTION) != REJECTION_SHA256:
        raise RuntimeError("VG063 source rejection changed")
    rejection = json.loads(REJECTION.read_text())
    if (
        rejection.get("schema")
        != "vq2_vg062_warmed_teacher_intervention_rejection_v1"
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("live_authority")
        or rejection.get("failed_admission_predicates")
        != ["minimum_success_rate", "all_later_public_heads_observed"]
    ):
        raise RuntimeError("VG062 does not authorize the corrected horizon")


def configure() -> None:
    core.TAG = TAG
    core.SCHEMA = REPORT_SCHEMA
    core.STATE_SCHEMA = STATE_SCHEMA
    core.SEED = SEED
    core.STEP_LIMIT = STEP_LIMIT
    core.MINIMUM_SUCCESS_RATE = MINIMUM_SUCCESS_RATE
    core.MINIMUM_RECORDS = MINIMUM_RECORDS
    core.PREREGISTRATION = PREREGISTRATION
    core.RUNNER = RUNNER
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), REJECTION, TEST)
    core.verify_inputs = verify_inputs


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False):
    configure()
    return core.collect(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["training_dataset_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
