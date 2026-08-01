#!/usr/bin/env python3
"""Recapture LC105 phase-8 outcomes in its exact full-batch context."""

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
import scripts.collect_vq2_lc100_phase7_full_batch_recapture as base


TAG = "vq2_lc106_phase8_full_batch_recapture_001"
SCHEMA = "vq2_lc106_phase8_full_batch_recapture_report_v1"
STATE_SCHEMA = "vq2_lc106_phase8_full_batch_recapture_state_v1"
SEED = 432_050
TARGET_PHASE = 8
MINIMUM_RECORDS = 4_000
EXPECTED_QUERY_AGENTS = 4
EXPECTED_SUCCESS_AGENTS = 2
EXPECTED_FAILURE_AGENTS = 2
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc105_phase7_endpoint_full_course_001"
)
CHECKPOINT = PARENT_DIR / "policy_selected.pt"
CHECKPOINT_SHA256 = "005e5e7929258fd282ab390fd230fda817700afaee790e78144200e67b3e10a4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c614e6929282399cac2f18465084f8337ed8770389a74e48b86ff1fcf4315e7d"
PREREGISTRATION = ROOT / "docs/vq2_lc106_phase8_full_batch_recapture_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc106_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc106_phase8_full_batch_recapture.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
CHECKPOINT_SCHEMA_EXPECTED = "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"


def verify_inputs() -> None:
    for path, digest in {
        CHECKPOINT: CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC106 bound input changed: {path}")
    parent = json.loads(PARENT_REPORT.read_text())
    selected = parent.get("selected_candidate", {})
    distribution = selected.get("maximum_raw_index_distribution", {})
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_report_v1"
        or not parent.get("diagnostic_valid")
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or parent.get("seed") != SEED
        or parent.get("group_size") != base.SEED_GROUP_SIZE
        or selected.get("mean_gates_passed") != 3.4296875
        or distribution.get("8") != 1
        or distribution.get("9") != 1
        or parent.get("safety", {}).get("flight_sim_packets_sent") != 0
        or parent.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105 does not authorize LC106")


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.SEED, base.TARGET_PHASE = SEED, TARGET_PHASE
    base.MINIMUM_RECORDS = MINIMUM_RECORDS
    base.EXPECTED_QUERY_AGENTS = EXPECTED_QUERY_AGENTS
    base.EXPECTED_SUCCESS_AGENTS = EXPECTED_SUCCESS_AGENTS
    base.EXPECTED_FAILURE_AGENTS = EXPECTED_FAILURE_AGENTS
    base.CHECKPOINT, base.CHECKPOINT_SHA256 = CHECKPOINT, CHECKPOINT_SHA256
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    base.LC099_REPORT, base.LC099_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    base.CHECKPOINT_SCHEMA_EXPECTED = CHECKPOINT_SCHEMA_EXPECTED
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    original = base.verify_inputs
    base.verify_inputs = verify_inputs
    try:
        base.configure()
        base.recapture.base.EXTRA_SOURCE_PATHS = (
            Path(__file__).resolve(), TEST, Path(base.__file__).resolve(),
            PARENT_REPORT,
        )
        return base.recapture.base.base.collect(
            output=output, device_name=device_name, resume=resume
        )
    finally:
        base.verify_inputs = original


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
