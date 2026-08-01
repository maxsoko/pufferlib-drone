#!/usr/bin/env python3
"""Recapture LC087 phase-7 outcomes with its exact paired seed mapping."""

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
import scripts.collect_vq2_lc091_phase7_success_recapture as recapture


TAG = "vq2_lc092_phase7_paired_seed_recapture_001"
SCHEMA = "vq2_lc092_phase7_paired_seed_recapture_report_v1"
STATE_SCHEMA = "vq2_lc092_phase7_paired_seed_recapture_state_v1"
AGENTS = EPISODES = 256
SEED_GROUP_SIZE = 128
MINIMUM_RECORDS = 2_000
EXPECTED_QUERY_AGENTS = 6
EXPECTED_SUCCESS_AGENTS = 4
EXPECTED_FAILURE_AGENTS = 2
PARENT_REPORT = recapture.PARENT_REPORT
PARENT_REPORT_SHA256 = recapture.PARENT_REPORT_SHA256
CHECKPOINT = recapture.CHECKPOINT
CHECKPOINT_SHA256 = recapture.CHECKPOINT_SHA256
LC091_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc091_phase7_success_recapture_001/report.json"
)
LC091_REPORT_SHA256 = "0c5d0630e80066337fbdb940ec062053cb7989023199f5b7e1baf8a58cc42efe"
PREREGISTRATION = ROOT / "docs/vq2_lc092_phase7_paired_seed_recapture_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc092_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc092_phase7_paired_seed_recapture.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    for path, digest in {
        CHECKPOINT: CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC091_REPORT: LC091_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC092 bound input changed: {path}")
    parent = json.loads(PARENT_REPORT.read_text())
    failed = json.loads(LC091_REPORT.read_text())
    selected = parent.get("selected_candidate", {})
    if (
        parent.get("schema") != "vq2_lc087_phase6_half_bias_full_course_report_v1"
        or not parent.get("numerically_admitted")
        or parent.get("seed") != recapture.SEED
        or parent.get("group_size") != SEED_GROUP_SIZE
        or parent.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or selected.get("maximum_raw_index_distribution", {}).get("7") != 1
        or selected.get("maximum_raw_index_distribution", {}).get("8") != 1
        or selected.get("maximum_raw_index_distribution", {}).get("9") != 1
        or failed.get("schema") != "vq2_lc091_phase7_success_recapture_report_v1"
        or failed.get("training_dataset_admitted")
        or failed.get("query_agents") != 0
        or failed.get("seed") != recapture.SEED
        or failed.get("safety", {}).get("flight_sim_packets_sent") != 0
        or failed.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC087/LC091 do not authorize corrected paired recapture")


def paired_dagger_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = recapture.base.dagger_config(pufferl_module)
    config["vec"]["env_seed_group_size"] = SEED_GROUP_SIZE
    return config, [*overrides, "--vec.env-seed-group-size", str(SEED_GROUP_SIZE)]


def configure() -> None:
    recapture.TAG, recapture.SCHEMA, recapture.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    recapture.AGENTS = recapture.EPISODES = AGENTS
    recapture.MINIMUM_RECORDS = MINIMUM_RECORDS
    recapture.EXPECTED_QUERY_AGENTS = EXPECTED_QUERY_AGENTS
    recapture.EXPECTED_SUCCESS_AGENTS = EXPECTED_SUCCESS_AGENTS
    recapture.EXPECTED_FAILURE_AGENTS = EXPECTED_FAILURE_AGENTS
    recapture.LC090_REPORT = LC091_REPORT
    recapture.LC090_REPORT_SHA256 = LC091_REPORT_SHA256
    recapture.PREREGISTRATION, recapture.RUNNER, recapture.TEST = (
        PREREGISTRATION, RUNNER, TEST,
    )
    recapture.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    recapture.verify_inputs = verify_inputs
    recapture.configure()
    recapture.base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), TEST, Path(recapture.__file__).resolve(),
        PARENT_REPORT, LC091_REPORT,
    )
    recapture.base.base.intervention_config = paired_dagger_config


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    configure()
    return recapture.base.base.collect(
        output=output, device_name=device_name, resume=resume
    )


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
