#!/usr/bin/env python3
"""Collect phase-7 states in the proven 1024-agent numerical layout."""

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
import scripts.collect_vq2_lc051_index7_capped_student_features as base


TAG = "vq2_lc052_index7_capped_student_features_001"
SCHEMA = "vq2_lc052_index7_capped_student_report_v1"
STATE_SCHEMA = "vq2_lc052_index7_capped_student_state_v1"
AGENTS = EPISODES = 1024
SEED = 431520
RECORD_STOP = MINIMUM_RECORDS = 10_000
MINIMUM_QUERY_AGENTS = 2
REJECTION = ROOT / "docs/vq2_lc051_batch2048_rejection_2026-07-31.json"
REJECTION_SHA256 = (
    "1278efe17a5dea5484eb37bb96543c33b43bc3be4b71f6c4aae5e736514f64d8"
)
LC051_STATE = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc051_index7_capped_student_features_001_state.json"
)
LC051_STATE_SHA256 = (
    "51d8d3b3719f7fea01b4b2a0dcd755399b75612e76a5250063b25802d2e2c263"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc052_index7_capped_student_features_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc052_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
BASE_VERIFY_INPUTS = base.verify_inputs


def verify_inputs() -> None:
    BASE_VERIFY_INPUTS()
    expected = {
        REJECTION: REJECTION_SHA256,
        LC051_STATE: LC051_STATE_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC052 bound rejection evidence changed: {path}")
    rejected = json.loads(REJECTION.read_text())
    state = json.loads(LC051_STATE.read_text())
    if (
        rejected.get("schema") != "vq2_lc051_batch2048_rejection_v1"
        or not rejected.get("rejected")
        or not rejected.get("unchanged_retry_forbidden")
        or rejected.get("feature_bytes") != 0
        or rejected.get("minimum_observed_wall_time_seconds", 0) < 440
        or rejected.get("state_sha256") != LC051_STATE_SHA256
        or state.get("schema") != "vq2_lc051_index7_capped_student_state_v1"
        or state.get("agents") != 2048
        or state.get("intervention_phase_min") != 7
        or state.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("submission_authorized")
    ):
        raise RuntimeError("LC051 does not authorize the 1024-agent recovery")


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.AGENTS, base.EPISODES, base.SEED = AGENTS, EPISODES, SEED
    base.RECORD_STOP = base.MINIMUM_RECORDS = RECORD_STOP
    base.MINIMUM_QUERY_AGENTS = MINIMUM_QUERY_AGENTS
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.verify_inputs = verify_inputs
    base.configure()
    base.base.EXTRA_SOURCE_PATHS = (
        *base.base.EXTRA_SOURCE_PATHS,
        Path(__file__).resolve(), REJECTION, LC051_STATE,
    )


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
            resume: bool = False) -> dict[str, Any]:
    configure()
    return base.base.base.collect(
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
