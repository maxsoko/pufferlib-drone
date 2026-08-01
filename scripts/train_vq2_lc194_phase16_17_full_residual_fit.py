#!/usr/bin/env python3
"""Fit LC189's phase-16/17 residual heads on the admitted LC193 rescue corpus."""

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
import scripts.train_vq2_lc111_phase8_9_full_residual_endpoint as prior


TAG = "vq2_lc194_phase16_17_full_residual_fit_001"
SCHEMA = "vq2_lc194_phase16_17_full_residual_fit_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc194_phase16_17_full_residual_fit_checkpoint_v1"
SEED = 432_194
PHASES = (16, 17)
OPTIMIZER_STEPS = 512
BATCH_SIZE = 4_096
EVALUATION_INTERVAL = 16
LEARNING_RATE = 3e-3
ANCHOR_COEFFICIENT = 1e-3
GRADIENT_CLIP = 1.0
TARGET_ACTION_CLIP = 0.999
MINIMUM_PHASE_IMPROVEMENT = 2.0
MAXIMUM_PHASE_DELTA_L2 = 64.0
PARENT_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc189_phase16_adapter_split_batch_cem_001"
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "f9c2ee5a5db07ba911470e7e87fe2016a4bf0c7ee9250cb3bd6b07427e1c9c21"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "944be397e95d2bcc427e99d09e0b7885f1b0f1e7f2c2bacaf51a6e10398fc0b2"
DATASET_DIR = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc193_phase16_17_rescue_dagger_001"
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "699bd082aec4e0d6471377949639ed48c69329f39e47f28634d1c3ec5c739a81"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "50b1a311a153dc0277fa2ba85d97e043cd31b022aa01f5b519acd0df83c10b64"
PREREGISTRATION = ROOT / "docs/vq2_lc194_phase16_17_full_residual_fit_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc194_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc194_phase16_17_full_residual_fit.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC194 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    phase_records = dataset.get("feature_phase_records", [])
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent.get("frozen_non_phase16_state_exact")
        or parent_report.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema") != "vq2_lc193_phase16_17_rescue_dagger_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_records") != 715_520
        or len(phase_records) != 33
        or phase_records[16] != 454_656
        or phase_records[17] != 260_864
        or dataset.get("query_outcome_success_agents") != list(range(256, 512))
        or dataset.get("query_outcome_failure_agents") != list(range(256))
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC193 do not authorize LC194")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc193_phase16_17_rescue_dagger.py",
        ROOT / "scripts/train_vq2_lc111_phase8_9_full_residual_endpoint.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
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


def configure() -> tuple[tuple[str, ...], tuple[Any, ...]]:
    names = (
        "TAG", "SCHEMA", "CHECKPOINT_SCHEMA", "SEED", "PHASES",
        "OPTIMIZER_STEPS", "BATCH_SIZE", "EVALUATION_INTERVAL",
        "LEARNING_RATE", "ANCHOR_COEFFICIENT", "GRADIENT_CLIP",
        "TARGET_ACTION_CLIP", "MINIMUM_PHASE_IMPROVEMENT",
        "MAXIMUM_PHASE_DELTA_L2", "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256", "PARENT_REPORT", "PARENT_REPORT_SHA256",
        "FEATURES", "FEATURES_SHA256", "DATASET_REPORT",
        "DATASET_REPORT_SHA256", "PREREGISTRATION", "RUNNER", "TEST",
        "DEFAULT_OUTPUT", "verify_inputs", "source_identity",
    )
    originals = tuple(getattr(prior, name) for name in names)
    values = (
        TAG, SCHEMA, CHECKPOINT_SCHEMA, SEED, PHASES,
        OPTIMIZER_STEPS, BATCH_SIZE, EVALUATION_INTERVAL,
        LEARNING_RATE, ANCHOR_COEFFICIENT, GRADIENT_CLIP,
        TARGET_ACTION_CLIP, MINIMUM_PHASE_IMPROVEMENT,
        MAXIMUM_PHASE_DELTA_L2, PARENT_CHECKPOINT,
        PARENT_CHECKPOINT_SHA256, PARENT_REPORT, PARENT_REPORT_SHA256,
        FEATURES, FEATURES_SHA256, DATASET_REPORT,
        DATASET_REPORT_SHA256, PREREGISTRATION, RUNNER, TEST,
        DEFAULT_OUTPUT, verify_inputs, source_identity,
    )
    for name, value in zip(names, values):
        setattr(prior, name, value)
    return names, originals


def restore(snapshot: tuple[tuple[str, ...], tuple[Any, ...]]) -> None:
    names, originals = snapshot
    for name, value in zip(names, originals):
        setattr(prior, name, value)


def fit(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    verify_inputs()
    snapshot = configure()
    try:
        return prior.fit(output=output, device_name=device_name, resume=resume)
    finally:
        restore(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = fit(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("numerically_admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
