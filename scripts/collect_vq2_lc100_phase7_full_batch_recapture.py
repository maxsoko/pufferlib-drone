#!/usr/bin/env python3
"""Recapture LC094 phase-7 outcomes in its exact full-batch seed context."""

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


TAG = "vq2_lc100_phase7_full_batch_recapture_001"
SCHEMA = "vq2_lc100_phase7_full_batch_recapture_report_v1"
STATE_SCHEMA = "vq2_lc100_phase7_full_batch_recapture_state_v1"
AGENTS = EPISODES = 256
SEED = 431_990
SEED_GROUP_SIZE = 128
MINIMUM_RECORDS = 10_000
EXPECTED_QUERY_AGENTS = 8
EXPECTED_SUCCESS_AGENTS = 2
EXPECTED_FAILURE_AGENTS = 6
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc094_full_batch_serialized_checkpoint_001"
)
CHECKPOINT = PARENT_DIR / "policy_selected.pt"
CHECKPOINT_SHA256 = "14a5f90e8a7a80d3535ef0d86d3f9b5d75d42317c387a0554cf59ecf283e340a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "639edd98ec0d99c981d16837c802d758adc6622736314b3e6afe09a58af21139"
LC099_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc099_late_phase_safe_alpha_full_course_001/report.json"
)
LC099_REPORT_SHA256 = "385fc1ae3ef61937d56951dfd776f5d1cd5fe7cdc4a1cf99d7a17ffff2ff3e51"
PREREGISTRATION = ROOT / "docs/vq2_lc100_phase7_full_batch_recapture_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc100_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc100_phase7_full_batch_recapture.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> None:
    for path, digest in {
        CHECKPOINT: CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC099_REPORT: LC099_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC100 bound input changed: {path}")
    parent = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC099_REPORT.read_text())
    parent_item, candidate_item = rejected.get("items", [{}, {}])
    distribution = parent_item.get("maximum_raw_index_distribution", {})
    if (
        parent.get("schema") != "vq2_lc094_full_batch_serialized_checkpoint_report_v1"
        or not parent.get("diagnostic_valid")
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or rejected.get("schema") != "vq2_lc099_late_phase_safe_alpha_full_course_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("numerically_admitted")
        or rejected.get("seed") != SEED
        or rejected.get("group_size") != SEED_GROUP_SIZE
        or rejected.get("actor_execution_contract")
        != "two independently loaded complete saved Puffer checkpoints, each executed over the full 256-agent batch"
        or parent_item.get("mean_gates_passed") != 3.421875
        or distribution.get("7") != 3
        or distribution.get("8") != 0
        or distribution.get("9") != 1
        or candidate_item.get("mean_gates_passed") != 3.390625
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC094/LC099 do not authorize LC100")


def paired_dagger_config(pufferl_module: Any) -> tuple[dict[str, Any], list[str]]:
    config, overrides = recapture.base.dagger_config(pufferl_module)
    config["vec"]["env_seed_group_size"] = SEED_GROUP_SIZE
    return config, [*overrides, "--vec.env-seed-group-size", str(SEED_GROUP_SIZE)]


def configure() -> None:
    recapture.TAG, recapture.SCHEMA, recapture.STATE_SCHEMA = TAG, SCHEMA, STATE_SCHEMA
    recapture.AGENTS = recapture.EPISODES = AGENTS
    recapture.SEED = SEED
    recapture.MINIMUM_RECORDS = MINIMUM_RECORDS
    recapture.EXPECTED_QUERY_AGENTS = EXPECTED_QUERY_AGENTS
    recapture.EXPECTED_SUCCESS_AGENTS = EXPECTED_SUCCESS_AGENTS
    recapture.EXPECTED_FAILURE_AGENTS = EXPECTED_FAILURE_AGENTS
    recapture.CHECKPOINT, recapture.CHECKPOINT_SHA256 = CHECKPOINT, CHECKPOINT_SHA256
    recapture.PARENT_REPORT, recapture.PARENT_REPORT_SHA256 = (
        PARENT_REPORT, PARENT_REPORT_SHA256,
    )
    recapture.LC090_REPORT, recapture.LC090_REPORT_SHA256 = (
        LC099_REPORT, LC099_REPORT_SHA256,
    )
    recapture.PREREGISTRATION, recapture.RUNNER, recapture.TEST = (
        PREREGISTRATION, RUNNER, TEST,
    )
    recapture.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    recapture.verify_inputs = verify_inputs
    recapture.configure()
    recapture.base.CHECKPOINT_SCHEMA_EXPECTED = (
        "vq2_lc094_full_batch_serialized_checkpoint_checkpoint_v1"
    )
    recapture.base.EXTRA_SOURCE_PATHS = (
        Path(__file__).resolve(), TEST, Path(recapture.__file__).resolve(),
        PARENT_REPORT, LC099_REPORT,
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
