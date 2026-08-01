#!/usr/bin/env python3
"""Collect paired LC123 phase-8 Puffer anchors and oracle rescues."""

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
import scripts.collect_vq2_lc125_lc123_phase8_9_rescue_features as base


BASE_WRITE_JSON_ONCE = base.write_json_once

TAG = "vq2_lc134_phase8_paired_anchor_rescue_001"
SCHEMA = "vq2_lc134_phase8_paired_anchor_rescue_report_v1"
FEATURE_SCHEMA = "vq2_lc134_phase8_paired_anchor_rescue_feature_v1"
TARGET_RAW_INDEX = 9
PHASE_MIN = 8
PHASE_MAX_EXCLUSIVE = 9
PARENT_CHECKPOINT = base.PARENT_CHECKPOINT
PARENT_CHECKPOINT_SHA256 = base.PARENT_CHECKPOINT_SHA256
PARENT_REPORT = base.PARENT_REPORT
PARENT_REPORT_SHA256 = base.PARENT_REPORT_SHA256
LC133_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc133_phase8_anisotropic_ppo_001/report.json"
)
LC133_REPORT_SHA256 = "d2dfb18052993eccd202de211dec0396927a7bdeaf9c74424f4615e6b61be622"
PREREGISTRATION = ROOT / "docs/vq2_lc134_phase8_paired_anchor_rescue_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc134_vast.sh"
TEST = ROOT / "tests/test_collect_vq2_lc134_phase8_paired_anchor_rescue.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        LC133_REPORT: LC133_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC134 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    rejected = json.loads(LC133_REPORT.read_text())
    rollouts = rejected.get("rollouts", [])
    if (
        parent.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or rejected.get("schema") != "vq2_lc133_phase8_anisotropic_ppo_report_v1"
        or rejected.get("candidate_selected_for_screen") is not None
        or len(rollouts) != 4
        or any(item.get("raw9_or_later") != 0 for item in rollouts)
        or rollouts[-1].get("queried_phase_return_mean", -1e9)
        <= rollouts[0].get("queried_phase_return_mean", 1e9)
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC123/LC133 do not authorize LC134")
    return parent


def source_identity() -> dict[str, Any]:
    from pufferlib import _C

    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, LC133_REPORT,
        ROOT / "scripts/collect_vq2_lc125_lc123_phase8_9_rescue_features.py",
        ROOT / "pufferlib/vq2_informed.py", ROOT / "pufferlib/vq2_oracle.py",
        ROOT / "pufferlib/vq2_public_phase.py", ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
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
        success = corrected["query_outcome_success_agents"]
        failure = corrected["query_outcome_failure_agents"]
        predicates = {
            "paired_phase8_rescue": bool(
                control["target_passes"] == 0
                and intervention["target_passes"] == 3
                and intervention["paired_target_gains_vs_control"] == 3
                and intervention["paired_target_losses_vs_control"] == 0
            ),
            "paired_splittable_outcomes": bool(
                len(success) == 3 and len(failure) == 3
                and all(agent >= base.GROUP_SIZE for agent in success)
                and all(agent < base.GROUP_SIZE for agent in failure)
            ),
            "both_groups_recorded": bool(
                corrected["query_agents"] == 6
                and corrected["feature_records"]
                > intervention["offline_teacher_plant_actions"]
            ),
            "only_phase8": bool(
                corrected["feature_phase_records"][PHASE_MIN]
                == corrected["feature_records"]
            ),
            "finite_in_envelope_features": bool(
                corrected["teacher_action_envelope_violations"] == 0
            ),
            "transport_and_pairing": bool(
                corrected["diagnostic_valid"]
                and corrected["initial_seed_groups_exact"]
            ),
            "training_only_authority": True,
        }
        corrected["feature_schema"] = FEATURE_SCHEMA
        corrected["training_dataset_admitted"] = all(predicates.values())
        corrected["admission_predicates"] = predicates
        corrected["failed_admission_predicates"] = [
            name for name, passed in predicates.items() if not passed
        ]
        corrected["control_feature_target"] = "LC123 complete Puffer action anchor"
        corrected["intervention_feature_target"] = "training-only phase-8 oracle action"
        corrected["next_authority"] = (
            "Fit one source-locked phase-8 whole-Puffer residual endpoint; no FlightSim authority."
            if corrected["training_dataset_admitted"]
            else "Reject LC134 and retain LC105; do not fit or run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.FEATURE_SCHEMA,
        base.TARGET_RAW_INDEX, base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE,
        base.CAPTURE_CONTROL_FEATURES,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
    )
    base.TAG, base.SCHEMA, base.FEATURE_SCHEMA = TAG, SCHEMA, FEATURE_SCHEMA
    base.TARGET_RAW_INDEX = TARGET_RAW_INDEX
    base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE = PHASE_MIN, PHASE_MAX_EXCLUSIVE
    base.CAPTURE_CONTROL_FEATURES = True
    base.PREREGISTRATION, base.RUNNER, base.TEST = PREREGISTRATION, RUNNER, TEST
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.write_json_once = corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.FEATURE_SCHEMA,
        base.TARGET_RAW_INDEX, base.PHASE_MIN, base.PHASE_MAX_EXCLUSIVE,
        base.CAPTURE_CONTROL_FEATURES,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
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
    return 0 if report["training_dataset_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
