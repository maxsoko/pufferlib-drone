#!/usr/bin/env python3
"""Fit LC162 phase 15 on LC165 failure-state DAgger labels."""

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
import scripts.train_vq2_lc153_phase11_failure_state_dagger_fit as prior


TAG = "vq2_lc166_phase15_failure_state_dagger_fit_001"
SCHEMA = "vq2_lc166_phase15_failure_state_dagger_fit_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc166_phase15_failure_state_dagger_fit_checkpoint_v1"
TARGET_PHASE = 15
SUCCESS_AGENTS = tuple(range(512))
CONTROL_AGENTS = tuple(range(256))
TRAINING_SUCCESS_COUNT = 384
MAXIMUM_CONTROL_PARENT_DRIFT_MSE = 1.0
FROZEN_STATE_FIELD = "frozen_non_phase15_state_exact"
FIT_METADATA_FIELD = "phase15_failure_state_dagger_fit"
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc162_phase14_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "4a45d1814e2f76b776ed2a278bd9a026dab5c1caf346792bc26032eea874770f"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "a85ae9c0da3c52110891d03cf67eafeebf76a3123b62a210d3740e541d32fabe"
DATASET_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc165_phase15_failure_state_dagger_001"
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "f91e9c9a72d9122f46153ca424ec9bb38944b83ebc0fa9f15f330da9f57088d0"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "51a010be585552d4ce9a666f49a361f90412113e4886d5545db9ac70c99c210a"
FEATURE_RECORDS = 650_752
PREREGISTRATION = ROOT / "docs/vq2_lc166_phase15_failure_state_dagger_fit_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc166_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc166_phase15_failure_state_dagger_fit.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC166 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    control, intervention = dataset.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc162_phase14_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc162_phase14_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or dataset.get("schema") != "vq2_lc165_phase15_failure_state_dagger_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_records") != FEATURE_RECORDS
        or dataset.get("query_agents") != 512
        or dataset.get("query_outcome_success_agents") != list(range(256, 512))
        or dataset.get("query_outcome_failure_agents") != list(range(256))
        or not dataset.get("control_features_use_teacher_targets")
        or control.get("target_passes") != 0
        or intervention.get("target_passes") != 256
        or intervention.get("paired_target_losses_vs_control") != 0
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC162/LC165 do not authorize LC166")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc165_phase15_failure_state_dagger.py",
        ROOT / "scripts/train_vq2_lc153_phase11_failure_state_dagger_fit.py",
        ROOT / "scripts/train_vq2_lc136_phase8_success_only_fit.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
    )
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path) for path in paths
        },
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
            "numpy": np.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        corrected["dataset_schema"] = "vq2_lc165_phase15_failure_state_dagger_report_v1"
        corrected["whole_puffer_phase"] = TARGET_PHASE
        corrected["training_trajectory_scope"] = "labeled failure and oracle-rescue rows"
        corrected["next_authority"] = (
            "Run one teacher-free repeated-source LC162-versus-LC166 raw-16 screen; no FlightSim authority."
            if corrected.get("numerically_admitted") else
            "Reject LC166 and retain LC162 as the source frontier; do not run FlightSim."
        )
    prior.BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA, prior.TARGET_PHASE,
        prior.SUCCESS_AGENTS, prior.CONTROL_AGENTS, prior.TRAINING_SUCCESS_COUNT,
        prior.MAXIMUM_CONTROL_PARENT_DRIFT_MSE,
        prior.FROZEN_STATE_FIELD, prior.FIT_METADATA_FIELD,
        prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.FEATURES, prior.FEATURES_SHA256,
        prior.DATASET_REPORT, prior.DATASET_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.verify_inputs, prior.source_identity, prior.corrected_writer,
    )
    prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    prior.TARGET_PHASE = TARGET_PHASE
    prior.SUCCESS_AGENTS, prior.CONTROL_AGENTS = SUCCESS_AGENTS, CONTROL_AGENTS
    prior.TRAINING_SUCCESS_COUNT = TRAINING_SUCCESS_COUNT
    prior.MAXIMUM_CONTROL_PARENT_DRIFT_MSE = MAXIMUM_CONTROL_PARENT_DRIFT_MSE
    prior.FROZEN_STATE_FIELD, prior.FIT_METADATA_FIELD = FROZEN_STATE_FIELD, FIT_METADATA_FIELD
    prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    prior.FEATURES, prior.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    prior.DATASET_REPORT, prior.DATASET_REPORT_SHA256 = DATASET_REPORT, DATASET_REPORT_SHA256
    prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    prior.verify_inputs, prior.source_identity, prior.corrected_writer = (
        verify_inputs, source_identity, corrected_writer,
    )
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        prior.TAG, prior.SCHEMA, prior.CHECKPOINT_SCHEMA, prior.TARGET_PHASE,
        prior.SUCCESS_AGENTS, prior.CONTROL_AGENTS, prior.TRAINING_SUCCESS_COUNT,
        prior.MAXIMUM_CONTROL_PARENT_DRIFT_MSE,
        prior.FROZEN_STATE_FIELD, prior.FIT_METADATA_FIELD,
        prior.PARENT_CHECKPOINT, prior.PARENT_CHECKPOINT_SHA256,
        prior.PARENT_REPORT, prior.PARENT_REPORT_SHA256,
        prior.FEATURES, prior.FEATURES_SHA256,
        prior.DATASET_REPORT, prior.DATASET_REPORT_SHA256,
        prior.PREREGISTRATION, prior.RUNNER, prior.TEST, prior.DEFAULT_OUTPUT,
        prior.verify_inputs, prior.source_identity, prior.corrected_writer,
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
