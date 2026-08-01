#!/usr/bin/env python3
"""Fit a success-anchored phase-8 decoder endpoint from admitted LC106."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.fit_vq2_lc080_phase6_success_anchored_endpoint as base


TAG = "vq2_lc107_phase8_success_anchored_endpoint_001"
SCHEMA = "vq2_lc107_phase8_success_anchored_endpoint_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc107_phase8_success_anchored_endpoint_checkpoint_v1"
SEED = 432_070
TARGET_PHASE = 8
EXPECTED_QUERY_AGENTS = 4
EXPECTED_SUCCESS_AGENTS = 2
MINIMUM_FAILURE_IMPROVEMENT = 1.05
MAXIMUM_SUCCESS_ACTION_DRIFT_MSE = 0.00025
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc105_phase7_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "005e5e7929258fd282ab390fd230fda817700afaee790e78144200e67b3e10a4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c614e6929282399cac2f18465084f8337ed8770389a74e48b86ff1fcf4315e7d"
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc106_phase8_full_batch_recapture_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "e13fc384a159e92d0ef59733360d11c01d570fdd1044504c46793bde40e97a33"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "2bfa88c9caa53c4cde6a0ae51a79413f1c1bfe5e2f70f5e960b27b607f2b8999"
PREREGISTRATION = ROOT / "docs/vq2_lc107_phase8_success_anchored_endpoint_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc107_vast.sh"
TEST = ROOT / "tests/test_fit_vq2_lc107_phase8_success_anchored_endpoint.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC107 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc105_phase7_endpoint_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema") != "vq2_lc106_phase8_full_batch_recapture_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 4_736
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("query_agents") != EXPECTED_QUERY_AGENTS
        or dataset.get("query_outcome_success_agents") != EXPECTED_SUCCESS_AGENTS
        or dataset.get("query_outcome_failure_agents") != 2
        or dataset.get("query_outcome_censored_agents") != 0
        or dataset.get("teacher_plant_actions_executed") != 0
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC106 do not authorize LC107")
    return parent


def configure() -> None:
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.SEED, base.TARGET_PHASE = SEED, TARGET_PHASE
    base.EXPECTED_QUERY_AGENTS = EXPECTED_QUERY_AGENTS
    base.EXPECTED_SUCCESS_AGENTS = EXPECTED_SUCCESS_AGENTS
    base.MINIMUM_FAILURE_IMPROVEMENT = MINIMUM_FAILURE_IMPROVEMENT
    base.MAXIMUM_SUCCESS_ACTION_DRIFT_MSE = MAXIMUM_SUCCESS_ACTION_DRIFT_MSE
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.DATASET_REPORT, base.DATASET_REPORT_SHA256 = (
        DATASET_REPORT, DATASET_REPORT_SHA256,
    )
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.EXTRA_SOURCE_PATHS = (Path(__file__).resolve(), TEST)


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    original = base.verify_inputs
    base.verify_inputs = verify_inputs
    try:
        return base.fit(output=output, device_name=device_name, resume=resume)
    finally:
        base.verify_inputs = original


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
