#!/usr/bin/env python3
"""Horizon-corrected successor to the empty LC074 phase-6 probe."""

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
import scripts.collect_vq2_lc074_phase6_capped_student_features as base


TAG = "vq2_lc075_phase6_horizon_capped_features_001"
SCHEMA = "vq2_lc075_phase6_horizon_capped_features_report_v1"
STATE_SCHEMA = "vq2_lc075_phase6_horizon_capped_features_state_v1"
AGENTS = EPISODES = 256
THREADS = 32
SEED = 431750
TARGET_PHASE = 6
STEP_LIMIT = 12_000
RECORD_STOP = MINIMUM_RECORDS = 20_000
LC074_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc074_phase6_capped_student_features_001/report.json"
)
LC074_REPORT_SHA256 = "b57a81c4490704dd4ad2fbc3b9c4f26a35fc2d0188ea4ce45f4ca55f6425372d"
PREREGISTRATION = ROOT / "docs/vq2_lc075_phase6_capped_student_features_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc075_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc075_phase6_horizon_capped_features.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
_BASE_VERIFY_INPUTS = base.verify_inputs


def verify_inputs() -> None:
    _BASE_VERIFY_INPUTS()
    if sha256_path(LC074_REPORT) != LC074_REPORT_SHA256:
        raise RuntimeError("LC075 bound LC074 rejection changed")
    rejected = json.loads(LC074_REPORT.read_text())
    if (
        rejected.get("schema") != "vq2_lc074_phase6_capped_student_features_report_v1"
        or rejected.get("training_dataset_admitted")
        or rejected.get("feature_records") != 0
        or rejected.get("query_agents") != 0
        or rejected.get("vector_steps") != 6_000
        or rejected.get("completed_agents") != 256
        or rejected.get("executed_action_max_error") != 0.0
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC074 does not authorize the horizon correction")


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.AGENTS, base.EPISODES, base.THREADS = AGENTS, EPISODES, THREADS
    base.SEED, base.TARGET_PHASE = SEED, TARGET_PHASE
    base.STEP_LIMIT = STEP_LIMIT
    base.RECORD_STOP = base.MINIMUM_RECORDS = RECORD_STOP
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.verify_inputs = verify_inputs
    base.configure()
    base.base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), TEST, base.LC073_REPORT, LC074_REPORT,
    )


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    return base.base.base.collect(output=output, device_name=device_name, resume=resume)


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
