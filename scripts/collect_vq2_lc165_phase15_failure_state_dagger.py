#!/usr/bin/env python3
"""Collect phase-15 oracle labels on LC162 failure and rescue states."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc152_phase11_failure_state_dagger as prior


TAG = "vq2_lc165_phase15_failure_state_dagger_001"
SCHEMA = "vq2_lc165_phase15_failure_state_dagger_report_v1"
FEATURE_SCHEMA = "vq2_lc165_phase15_failure_state_dagger_feature_v1"
GROUP_SIZE = 256
TOTAL_AGENTS = 512
MAX_STEPS = 27_000
TARGET_RAW_INDEX = 16
PHASE_MIN = 15
PHASE_MAX_EXCLUSIVE = 16
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc162_phase14_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "4a45d1814e2f76b776ed2a278bd9a026dab5c1caf346792bc26032eea874770f"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "a85ae9c0da3c52110891d03cf67eafeebf76a3123b62a210d3740e541d32fabe"
LC164_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc164_phase15_split_batch_cem_001/report.json"
LC164_REPORT_SHA256 = "0dbc238fb6735a422cce5659edd54b3a7881aae2745f3a220210acbbfe5dcce1"
PREREGISTRATION = ROOT / "docs/vq2_lc165_phase15_failure_state_dagger_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc165_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc165_phase15_failure_state_dagger.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC164_REPORT: LC164_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC165 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    exhausted = json.loads(LC164_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc162_phase14_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc162_phase14_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or exhausted.get("schema") != "vq2_lc164_phase15_split_batch_cem_report_v1"
        or not exhausted.get("completed")
        or exhausted.get("training_admitted")
        or exhausted.get("candidate_selected_for_screen") is not None
        or len(exhausted.get("generations", [])) != 2
        or any(item.get("target_passes") != 0 for item in exhausted.get("generations", []))
        or any(item.get("query_agents") != 512 for item in exhausted.get("generations", []))
        or exhausted.get("safety", {}).get("flight_sim_packets_sent") != 0
        or exhausted.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC162/LC164 do not authorize LC165")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC164_REPORT,
        ROOT / "scripts/collect_vq2_lc152_phase11_failure_state_dagger.py",
        ROOT / "scripts/collect_vq2_lc147_phase11_state_dependent_rescue.py",
        ROOT / "scripts/collect_vq2_lc125_lc123_phase8_9_rescue_features.py",
        ROOT / "pufferlib/vq2_oracle.py", ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "ocean/drone_race/drone_race.c", ROOT / "ocean/drone_race/drone_race.h",
        ROOT / "src/vecenv.h", ROOT / "src/bindings.cu",
    )
    extension = Path(_C.__file__).resolve()
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {
            **{str(path.relative_to(ROOT)): sha256_path(path) for path in paths},
            "compiled_extension": sha256_path(extension),
        },
        "compiled_extension_path": str(extension),
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_dynamic": os.environ.get("OMP_DYNAMIC"),
        },
    }


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        control, intervention = corrected["items"]
        control["name"] = "lc162_puffer_failure_control"
        intervention["name"] = "lc162_phase15_alignment_oracle"
        success = corrected["query_outcome_success_agents"]
        failure = corrected["query_outcome_failure_agents"]
        predicates = {
            "state_dependent_rescue": bool(
                control["target_passes"] == 0
                and intervention["target_passes"] == GROUP_SIZE
                and intervention["paired_target_gains_vs_control"] == GROUP_SIZE
                and intervention["paired_target_losses_vs_control"] == 0
            ),
            "all_rows_query_phase15": corrected["query_agents"] == TOTAL_AGENTS,
            "split_success_failure_groups": bool(
                success == list(range(GROUP_SIZE, TOTAL_AGENTS))
                and failure == list(range(GROUP_SIZE))
            ),
            "failure_states_have_teacher_targets": bool(
                corrected.get("control_features_use_teacher_targets")
                and corrected["feature_records"]
                > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase15": bool(
                corrected["feature_phase_records"][PHASE_MIN]
                == corrected["feature_records"]
            ),
            "finite_in_envelope_features": (
                corrected["teacher_action_envelope_violations"] == 0
            ),
            "transport_and_pairing": bool(
                corrected["diagnostic_valid"] and corrected["initial_seed_groups_exact"]
            ),
            "training_only_authority": True,
        }
        corrected["feature_schema"] = FEATURE_SCHEMA
        corrected["training_dataset_admitted"] = all(predicates.values())
        corrected["admission_predicates"] = predicates
        corrected["failed_admission_predicates"] = [
            name for name, passed in predicates.items() if not passed
        ]
        corrected["actor_execution"] = (
            "two independent complete recurrent LC162 Puffer actors over 256 rows each"
        )
        corrected["next_authority"] = (
            "Fit one phase-15 whole-Puffer DAgger residual on both labeled failure and rescue trajectories; no FlightSim authority."
            if corrected["training_dataset_admitted"] else
            "Reject the LC165 DAgger dataset; do not fit or run FlightSim."
        )
    prior.BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        prior.TAG, prior.SCHEMA, prior.FEATURE_SCHEMA,
        prior.GROUP_SIZE, prior.TOTAL_AGENTS, prior.MAX_STEPS,
        prior.TARGET_RAW_INDEX, prior.PHASE_MIN, prior.PHASE_MAX_EXCLUSIVE,
        prior.ENV_SEED_GROUP_SIZE, prior.ENV_SEED_INDEX_OFFSET,
        prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.verify_inputs, prior.source_identity, prior.corrected_writer,
    )
    prior.TAG, prior.SCHEMA, prior.FEATURE_SCHEMA = TAG, SCHEMA, FEATURE_SCHEMA
    prior.GROUP_SIZE, prior.TOTAL_AGENTS, prior.MAX_STEPS = GROUP_SIZE, TOTAL_AGENTS, MAX_STEPS
    prior.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    prior.PHASE_MIN, prior.PHASE_MAX_EXCLUSIVE = PHASE_MIN, PHASE_MAX_EXCLUSIVE
    prior.ENV_SEED_GROUP_SIZE, prior.ENV_SEED_INDEX_OFFSET = (
        ENV_SEED_GROUP_SIZE, ENV_SEED_INDEX_OFFSET,
    )
    prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    prior.verify_inputs, prior.source_identity, prior.corrected_writer = (
        verify_inputs, source_identity, corrected_writer,
    )
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        prior.TAG, prior.SCHEMA, prior.FEATURE_SCHEMA,
        prior.GROUP_SIZE, prior.TOTAL_AGENTS, prior.MAX_STEPS,
        prior.TARGET_RAW_INDEX, prior.PHASE_MIN, prior.PHASE_MAX_EXCLUSIVE,
        prior.ENV_SEED_GROUP_SIZE, prior.ENV_SEED_INDEX_OFFSET,
        prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.verify_inputs, prior.source_identity, prior.corrected_writer,
    ) = originals


def collect(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        return prior.collect(output=output, device_name=device_name, resume=resume)
    finally:
        restore(originals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = collect(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("training_dataset_admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
