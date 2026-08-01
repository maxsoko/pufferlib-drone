#!/usr/bin/env python3
"""Fit LC179's adapter with early phase-15 rescue-entry weighting."""

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

from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseLocalAdapterActor
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_lc176_phase15_recurrent_adapter as prior


TAG = "vq2_lc180_phase15_recurrent_adapter_early_weighted_001"
SCHEMA = "vq2_lc180_phase15_recurrent_adapter_early_weighted_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc180_phase15_recurrent_adapter_early_weighted_checkpoint_v1"
EPOCHS = 100
LEARNING_RATE = 1e-3
MINIMUM_VALIDATION_IMPROVEMENT = 10.0
TEMPORAL_WEIGHT_SEGMENTS = (
    (0, 64, 8.0),
    (64, 128, 12.0),
    (128, 256, 6.0),
    (256, 481, 2.0),
)
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc179_phase15_recurrent_adapter_convergence_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "7cdb65bc308a29b4d1a42b2c0d1d1b3c94f73e124d5a3bbb54250fe57f325ade"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "cd6b98ed6803a12d709a5d049aa0ad9db9f07ceb11027837d7fbb00ac2e78c39"
LC178_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc178_phase15_recurrent_adapter_milestone_001/report.json"
LC178_REPORT_SHA256 = "644aa3d1e8155b57883243fe2e6cca42a38bd5bb4bb1db4e3151330a49e808b8"
PREREGISTRATION = ROOT / "docs/vq2_lc180_phase15_recurrent_adapter_early_weighted_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc180_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc180_phase15_recurrent_adapter_early_weighted.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
NEXT_AUTHORITY_ADMITTED = (
    "Run one teacher-free LC176-versus-LC180 raw-16 screen; no FlightSim authority."
)
NEXT_AUTHORITY_REJECTED = "Reject LC180; do not screen or run FlightSim."


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        prior.FEATURES: prior.FEATURES_SHA256,
        prior.DATASET_REPORT: prior.DATASET_REPORT_SHA256,
        LC178_REPORT: LC178_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC180 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(prior.DATASET_REPORT.read_text())
    rejected = json.loads(LC178_REPORT.read_text())
    history = parent_report.get("history", [])
    if (
        parent.get("schema") != "vq2_lc179_phase15_recurrent_adapter_convergence_checkpoint_v1"
        or parent.get("numerically_admitted")
        or parent.get("model", {}).get("adapter_target_phase") != prior.TARGET_PHASE
        or parent.get("model", {}).get("adapter_size") != prior.ADAPTER_SIZE
        or parent_report.get("schema") != "vq2_lc179_phase15_recurrent_adapter_convergence_report_v1"
        or parent_report.get("numerically_admitted")
        or not parent_report.get("frozen_base_state_exact")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or parent_report.get("best_epoch") != 80
        or parent_report.get("validation_improvement_factor") != 3.5860753441392297
        or parent_report.get("best_validation_teacher_action_mse") != 0.00037468601658474654
        or len(history) != 80
        or history[-1]["validation_teacher_action_mse"]
        >= history[0]["validation_teacher_action_mse"]
        or dataset.get("schema") != "vq2_lc172_phase15_failure_state_dagger3_report_v1"
        or not dataset.get("training_dataset_admitted")
        or rejected.get("schema") != "vq2_lc178_phase15_recurrent_adapter_milestone_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or any(item.get("target_passes") != 0 for item in rejected.get("items", []))
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC178/LC179 do not authorize LC180")
    return parent


def build_actor(parent: dict[str, Any]) -> VQ2PhaseLocalAdapterActor:
    contract = parent["model"]
    actor = VQ2PhaseLocalAdapterActor(
        target_phase=int(contract["adapter_target_phase"]),
        hidden_size=int(contract["hidden_size"]),
        residual_size=int(contract["residual_size"]),
        adapter_size=int(contract["adapter_size"]),
        initial_std=float(contract["initial_std"]),
    )
    actor.load_state_dict(parent["model_state"])
    return actor


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, prior.FEATURES, prior.DATASET_REPORT,
        LC178_REPORT, ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/train_vq2_lc176_phase15_recurrent_adapter.py",
    )
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_path(path) for path in paths},
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def configure() -> tuple[Any, ...]:
    originals = (
        prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA,
        prior.EPOCHS, prior.LEARNING_RATE, prior.MINIMUM_VALIDATION_IMPROVEMENT,
        prior.TEMPORAL_WEIGHT_SEGMENTS,
        prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.NEXT_AUTHORITY_ADMITTED, prior.NEXT_AUTHORITY_REJECTED,
        prior.verify_inputs, prior.source_identity, prior.build_actor,
    )
    prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    prior.EPOCHS, prior.LEARNING_RATE = EPOCHS, LEARNING_RATE
    prior.MINIMUM_VALIDATION_IMPROVEMENT = MINIMUM_VALIDATION_IMPROVEMENT
    prior.TEMPORAL_WEIGHT_SEGMENTS = TEMPORAL_WEIGHT_SEGMENTS
    prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    prior.NEXT_AUTHORITY_ADMITTED, prior.NEXT_AUTHORITY_REJECTED = NEXT_AUTHORITY_ADMITTED, NEXT_AUTHORITY_REJECTED
    prior.verify_inputs, prior.source_identity, prior.build_actor = verify_inputs, source_identity, build_actor
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA,
        prior.EPOCHS, prior.LEARNING_RATE, prior.MINIMUM_VALIDATION_IMPROVEMENT,
        prior.TEMPORAL_WEIGHT_SEGMENTS,
        prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.NEXT_AUTHORITY_ADMITTED, prior.NEXT_AUTHORITY_REJECTED,
        prior.verify_inputs, prior.source_identity, prior.build_actor,
    ) = originals


def fit(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        return prior.fit(output=output, device_name=device_name, resume=resume)
    finally:
        restore(originals)


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
