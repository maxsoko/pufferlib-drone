#!/usr/bin/env python3
"""Collect phase-11 oracle labels on LC148 failure and rescue states."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.collect_vq2_lc147_phase11_state_dependent_rescue as prior
import scripts.collect_vq2_lc125_lc123_phase8_9_rescue_features as base


BASE_WRITE_JSON_ONCE = base.write_json_once
BASE_ACTOR_LOADER = base.milestone.load_actor

TAG = "vq2_lc152_phase11_failure_state_dagger_001"
SCHEMA = "vq2_lc152_phase11_failure_state_dagger_report_v1"
FEATURE_SCHEMA = "vq2_lc152_phase11_failure_state_dagger_feature_v1"
GROUP_SIZE = 256
TOTAL_AGENTS = 512
MAX_STEPS = 17_000
TARGET_RAW_INDEX = 12
PHASE_MIN = 11
PHASE_MAX_EXCLUSIVE = 12
ENV_SEED_GROUP_SIZE = 1
ENV_SEED_INDEX_OFFSET = 15
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc148_phase11_state_dependent_fit_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "1a3fe9762f3cd2884aefc88c2e70eb69225d8d21bda9e9a0134e30d021a18b58"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "bb9733ac17a600cdfed4e35144e00c6233d3a7e4daeebd053e7bbd86c7b23e41"
LC150_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc150_phase11_repeated_source_milestone_001/report.json"
)
LC150_REPORT_SHA256 = "a91fc9048d0853efc85cf6068454de34dea1d90ef658aecb27633323a3e913af"
LC151_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc151_phase11_fit_endpoint_bracket_001/report.json"
)
LC151_REPORT_SHA256 = "302ba4178c576a3544f03b783c801fd8872defef1735a80ee13aa71172981e56"
PREREGISTRATION = ROOT / "docs/vq2_lc152_phase11_failure_state_dagger_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc152_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc152_phase11_failure_state_dagger.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC150_REPORT: LC150_REPORT_SHA256,
        LC151_REPORT: LC151_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC152 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    repeated = json.loads(LC150_REPORT.read_text())
    exhausted = json.loads(LC151_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc148_phase11_state_dependent_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc148_phase11_state_dependent_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or repeated.get("schema") != "vq2_lc150_phase11_repeated_source_milestone_report_v1"
        or not repeated.get("diagnostic_valid")
        or repeated.get("items", [{}, {}])[1].get("target_passes") != 0
        or exhausted.get("schema") != "vq2_lc151_phase11_fit_endpoint_bracket_report_v1"
        or not exhausted.get("diagnostic_valid")
        or exhausted.get("causal_screen_selected") is not None
        or any(item.get("target_passes") != 0 for item in exhausted.get("items", []))
        or exhausted.get("safety", {}).get("flight_sim_packets_sent") != 0
        or exhausted.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC148/LC150/LC151 do not authorize LC152")
    return parent


def split_actor_loader(payload: dict[str, Any], device: torch.device) -> prior.SplitBatchActor:
    return prior.SplitBatchActor((BASE_ACTOR_LOADER(payload, device), BASE_ACTOR_LOADER(payload, device)))


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC150_REPORT, LC151_REPORT,
        ROOT / "scripts/collect_vq2_lc125_lc123_phase8_9_rescue_features.py",
        ROOT / "scripts/collect_vq2_lc147_phase11_state_dependent_rescue.py",
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
        control["name"] = "lc148_puffer_failure_control"
        intervention["name"] = "lc148_phase11_alignment_oracle"
        success = corrected["query_outcome_success_agents"]
        failure = corrected["query_outcome_failure_agents"]
        predicates = {
            "state_dependent_rescue": bool(
                control["target_passes"] == 0
                and intervention["target_passes"] == GROUP_SIZE
                and intervention["paired_target_gains_vs_control"] == GROUP_SIZE
                and intervention["paired_target_losses_vs_control"] == 0
            ),
            "all_rows_query_phase11": corrected["query_agents"] == TOTAL_AGENTS,
            "split_success_failure_groups": bool(
                success == list(range(GROUP_SIZE, TOTAL_AGENTS))
                and failure == list(range(GROUP_SIZE))
            ),
            "failure_states_have_teacher_targets": bool(
                corrected.get("control_features_use_teacher_targets")
                and corrected["feature_records"]
                > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase11": bool(
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
            "two independent complete recurrent LC148 Puffer actors over 256 rows each"
        )
        corrected["next_authority"] = (
            "Fit one DAgger phase-11 Puffer residual on both labeled failure and rescue trajectories; no FlightSim authority."
            if corrected["training_dataset_admitted"] else
            "Reject the LC152 DAgger dataset; do not fit or run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.FEATURE_SCHEMA,
        base.GROUP_SIZE, base.TOTAL_AGENTS, base.MAX_STEPS,
        base.TARGET_RAW_INDEX, base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE,
        base.ENV_SEED_GROUP_SIZE, base.ENV_SEED_INDEX_OFFSET,
        base.CAPTURE_CONTROL_FEATURES, base.CAPTURE_CONTROL_TEACHER_TARGETS,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor,
    )
    base.TAG, base.SCHEMA, base.FEATURE_SCHEMA = TAG, SCHEMA, FEATURE_SCHEMA
    base.GROUP_SIZE, base.TOTAL_AGENTS, base.MAX_STEPS = GROUP_SIZE, TOTAL_AGENTS, MAX_STEPS
    base.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE = PHASE_MIN, PHASE_MAX_EXCLUSIVE
    base.ENV_SEED_GROUP_SIZE, base.ENV_SEED_INDEX_OFFSET = (
        ENV_SEED_GROUP_SIZE, ENV_SEED_INDEX_OFFSET,
    )
    base.CAPTURE_CONTROL_FEATURES = True
    base.CAPTURE_CONTROL_TEACHER_TARGETS = True
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    base.verify_inputs = verify_inputs
    base.source_identity = source_identity
    base.write_json_once, base.milestone.load_actor = corrected_writer, split_actor_loader
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.FEATURE_SCHEMA,
        base.GROUP_SIZE, base.TOTAL_AGENTS, base.MAX_STEPS,
        base.TARGET_RAW_INDEX, base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE,
        base.ENV_SEED_GROUP_SIZE, base.ENV_SEED_INDEX_OFFSET,
        base.CAPTURE_CONTROL_FEATURES, base.CAPTURE_CONTROL_TEACHER_TARGETS,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
        base.milestone.load_actor,
    ) = originals


def collect(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        base.collect(output=output, device_name=device_name, resume=resume)
        return json.loads((output / "report.json").read_text())
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
    return 0 if report["diagnostic_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
