#!/usr/bin/env python3
"""Fit LC148 phase 11 on LC152 failure-state DAgger labels."""

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
import scripts.train_vq2_lc136_phase8_success_only_fit as base


BASE_WRITE_JSON_ONCE = base.write_json_once

TAG = "vq2_lc153_phase11_failure_state_dagger_fit_001"
SCHEMA = "vq2_lc153_phase11_failure_state_dagger_fit_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc153_phase11_failure_state_dagger_fit_checkpoint_v1"
TARGET_PHASE = 11
SUCCESS_AGENTS = tuple(range(512))
CONTROL_AGENTS = tuple(range(256))
TRAINING_SUCCESS_COUNT = 384
MAXIMUM_CONTROL_PARENT_DRIFT_MSE = 1.0
FROZEN_STATE_FIELD = "frozen_non_phase11_state_exact"
FIT_METADATA_FIELD = "phase11_failure_state_dagger_fit"
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc148_phase11_state_dependent_fit_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "1a3fe9762f3cd2884aefc88c2e70eb69225d8d21bda9e9a0134e30d021a18b58"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "bb9733ac17a600cdfed4e35144e00c6233d3a7e4daeebd053e7bbd86c7b23e41"
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc152_phase11_failure_state_dagger_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "72ff6b755ed13859719e4a99c408d108bc44df061c4985f11a998a53c0dc2dda"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "fa9720fc6b21ce4591e12de5276c75c651cf95f4a01d34232ad3c36bb4325be9"
PREREGISTRATION = ROOT / "docs/vq2_lc153_phase11_failure_state_dagger_fit_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc153_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc153_phase11_failure_state_dagger_fit.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC153 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    control, intervention = dataset.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc148_phase11_state_dependent_fit_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc148_phase11_state_dependent_fit_report_v1"
        or not parent_report.get("numerically_admitted")
        or dataset.get("schema") != "vq2_lc152_phase11_failure_state_dagger_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_records") != 318_464
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
        raise RuntimeError("LC148/LC152 do not authorize LC153")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc152_phase11_failure_state_dagger.py",
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
        corrected["dataset_schema"] = "vq2_lc152_phase11_failure_state_dagger_report_v1"
        corrected["whole_puffer_phase"] = TARGET_PHASE
        corrected["training_trajectory_scope"] = "labeled failure and oracle-rescue rows"
        corrected["next_authority"] = (
            "Run one teacher-free repeated-source LC148-versus-LC153 raw-12 screen; no FlightSim authority."
            if corrected.get("numerically_admitted") else
            "Reject LC153 and retain LC143 as the late-phase offline frontier; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA, base.TARGET_PHASE,
        base.SUCCESS_AGENTS, base.CONTROL_AGENTS, base.TRAINING_SUCCESS_COUNT,
        base.MAXIMUM_CONTROL_PARENT_DRIFT_MSE,
        base.FROZEN_STATE_FIELD, base.FIT_METADATA_FIELD,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.FEATURES, base.FEATURES_SHA256,
        base.DATASET_REPORT, base.DATASET_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
    )
    base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA = TAG, SCHEMA, CHECKPOINT_SCHEMA
    base.TARGET_PHASE = TARGET_PHASE
    base.SUCCESS_AGENTS, base.CONTROL_AGENTS = SUCCESS_AGENTS, CONTROL_AGENTS
    base.TRAINING_SUCCESS_COUNT = TRAINING_SUCCESS_COUNT
    base.MAXIMUM_CONTROL_PARENT_DRIFT_MSE = MAXIMUM_CONTROL_PARENT_DRIFT_MSE
    base.FROZEN_STATE_FIELD, base.FIT_METADATA_FIELD = (
        FROZEN_STATE_FIELD, FIT_METADATA_FIELD,
    )
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = (
        PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256,
    )
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.DATASET_REPORT, base.DATASET_REPORT_SHA256 = DATASET_REPORT, DATASET_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = (
        PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT,
    )
    base.verify_inputs, base.source_identity = verify_inputs, source_identity
    base.write_json_once = corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA, base.TARGET_PHASE,
        base.SUCCESS_AGENTS, base.CONTROL_AGENTS, base.TRAINING_SUCCESS_COUNT,
        base.MAXIMUM_CONTROL_PARENT_DRIFT_MSE,
        base.FROZEN_STATE_FIELD, base.FIT_METADATA_FIELD,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.FEATURES, base.FEATURES_SHA256,
        base.DATASET_REPORT, base.DATASET_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
    ) = originals


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    verify_inputs()
    originals = configure()
    try:
        base.fit(output=output, device_name=device_name, resume=resume)
        return json.loads((output / "report.json").read_text())
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
