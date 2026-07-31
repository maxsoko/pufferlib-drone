#!/usr/bin/env python3
"""Collect phase-7 states with 2048 envs and stable 1024-row actor chunks."""

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


TAG = "vq2_lc055_index7_chunked_student_features_001"
SCHEMA = "vq2_lc055_index7_chunked_student_report_v1"
STATE_SCHEMA = "vq2_lc055_index7_chunked_student_state_v1"
AGENTS = EPISODES = 2048
SEED = 431550
ACTOR_INFERENCE_CHUNK_SIZE = 1024
RECORD_STOP = MINIMUM_RECORDS = 20_000
MINIMUM_QUERY_AGENTS = 10
LC052_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc052_index7_capped_student_features_001/report.json"
)
LC052_REPORT_SHA256 = (
    "66e81b3df80ca6e8fe99551c0781b92805187d1f741feb6ff268f415cdcace2d"
)
LC054_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc054_phase7_equal_trajectory_head_001/report.json"
)
LC054_REPORT_SHA256 = (
    "a460280661e0e18d7e9ae28f156c290ba618ecfcfd168a08efb954cf0abe7372"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc055_index7_chunked_student_features_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc055_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
BASE_VERIFY_INPUTS = base.verify_inputs
BASE_PREDICATES = base.predicates


def verify_inputs() -> None:
    BASE_VERIFY_INPUTS()
    expected = {
        LC052_REPORT: LC052_REPORT_SHA256,
        LC054_REPORT: LC054_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC055 bound evidence changed: {path}")
    stable = json.loads(LC052_REPORT.read_text())
    rejected = json.loads(LC054_REPORT.read_text())
    if (
        stable.get("schema") != "vq2_lc052_index7_capped_student_report_v1"
        or not stable.get("training_dataset_admitted")
        or stable.get("agents") != 1024
        or stable.get("query_agents") != 6
        or stable.get("feature_records") != 10_000
        or stable.get("wall_time_seconds") != 408.9140789299272
        or rejected.get("schema")
        != "vq2_lc054_phase7_equal_trajectory_head_report_v1"
        or rejected.get("numerically_admitted")
        or rejected.get("equal_agent_weighting") is not True
        or rejected.get("selected_validation", {}).get("phases", {}).get(
            "7", {}
        ).get("improvement_factor") != 0.9964664111232708
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC052/LC054 do not authorize LC055")


def predicates(report: dict[str, Any]) -> dict[str, bool]:
    result = BASE_PREDICATES(report)
    result["fixed_stable_actor_inference_chunks"] = (
        report.get("actor_inference_chunk_size") == ACTOR_INFERENCE_CHUNK_SIZE
    )
    return result


def configure() -> None:
    base.TAG, base.SCHEMA, base.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    base.AGENTS, base.EPISODES, base.SEED = AGENTS, EPISODES, SEED
    base.RECORD_STOP = base.MINIMUM_RECORDS = RECORD_STOP
    base.MINIMUM_QUERY_AGENTS = MINIMUM_QUERY_AGENTS
    base.PREREGISTRATION, base.RUNNER = PREREGISTRATION, RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.verify_inputs = verify_inputs
    base.predicates = predicates
    base.configure()
    base.base.EXTRA_SOURCE_PATHS = (
        *base.base.EXTRA_SOURCE_PATHS,
        Path(__file__).resolve(), LC052_REPORT, LC054_REPORT,
    )
    base.base.base.ACTOR_INFERENCE_CHUNK_SIZE = ACTOR_INFERENCE_CHUNK_SIZE
    base.base.base.collection_predicates = predicates


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
