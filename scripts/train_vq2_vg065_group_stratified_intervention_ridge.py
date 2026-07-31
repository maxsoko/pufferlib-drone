#!/usr/bin/env python3
"""Run the group-stratified successor to the invalid VG064 holdout fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_vg064_indexed_intervention_ridge as core


BASE_VERIFY_INPUTS = core.verify_inputs
TAG = "vq2_vg065_group_stratified_intervention_ridge_001"
SCHEMA = "vq2_vg065_group_stratified_intervention_ridge_report_v1"
STATE_SCHEMA = "vq2_vg065_group_stratified_intervention_ridge_state_v1"
CHECKPOINT_SCHEMA = "vq2_vg065_group_stratified_intervention_ridge_checkpoint_v1"
SEED = 429200
FAILURE = ROOT / "docs/vq2_vg064_validation_split_infrastructure_rejection_2026-07-31.json"
FAILURE_SHA256 = "a38ca55065a822ec8f4f2e480be504c512061d1df8dab7164bd2d34c83d648da"
PREREGISTRATION = ROOT / "docs/vq2_vg065_group_stratified_intervention_ridge_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg065_vast.sh"
TEST = ROOT / "tests/test_train_vq2_vg065_group_stratified_intervention_ridge.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs():
    report = BASE_VERIFY_INPUTS()
    if sha256_path(FAILURE) != FAILURE_SHA256:
        raise RuntimeError("VG065 source failure evidence changed")
    failure = json.loads(FAILURE.read_text())
    if (
        failure.get("schema")
        != "vq2_vg064_validation_split_infrastructure_rejection_v1"
        or not failure.get("terminally_rejected")
        or not failure.get("unchanged_retry_forbidden")
        or failure.get("artifact_status", {}).get("checkpoint_written")
        or failure.get("live_authority")
    ):
        raise RuntimeError("VG064 does not authorize VG065")
    return report


def configure() -> None:
    core.TAG = TAG
    core.SCHEMA = SCHEMA
    core.STATE_SCHEMA = STATE_SCHEMA
    core.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    core.SEED = SEED
    core.PREREGISTRATION = PREREGISTRATION
    core.RUNNER = RUNNER
    core.TEST = TEST
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), FAILURE)
    core.verify_inputs = verify_inputs


def fit(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False):
    configure()
    return core.fit(output=output, device_name=device_name, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = fit(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
