#!/usr/bin/env python3
"""Fit phase 16 on LC193 rescue states while anchoring LC189 controls."""

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
TAG = "vq2_lc199_phase16_success_anchor_fit_001"
SCHEMA = "vq2_lc199_phase16_success_anchor_fit_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc199_phase16_success_anchor_fit_checkpoint_v1"
TARGET_PHASE = 16
SUCCESS_AGENTS = tuple(range(256, 512))
CONTROL_AGENTS = tuple(range(256))
TRAINING_SUCCESS_COUNT = 192
OPTIMIZER_STEPS = 512
LEARNING_RATE = 1e-3
SCALES = (0.003, 0.01, 0.03, 0.10, 0.30)
MINIMUM_SUCCESS_IMPROVEMENT = 1.01
MAXIMUM_CONTROL_PARENT_DRIFT_MSE = 2e-4
FROZEN_STATE_FIELD = "frozen_non_phase16_state_exact"
FIT_METADATA_FIELD = "phase16_success_anchor_fit"
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
LC198_REPORT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap/vq2_lc198_phase16_raw18_reroute_cem_001/report.json"
LC198_REPORT_SHA256 = "050fbc1b097ac03e5397adab9c52f7e8e11bc7d3e1dc39f7d6506f8954050c16"
PREREGISTRATION = ROOT / "docs/vq2_lc199_phase16_success_anchor_fit_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc199_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc199_phase16_success_anchor_fit.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        LC198_REPORT: LC198_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC199 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC198_REPORT.read_text())
    control, intervention = dataset.get("items", [{}, {}])
    if (
        parent.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc189_phase16_adapter_split_batch_cem_report_v1"
        or not parent_report.get("training_admitted")
        or dataset.get("schema") != "vq2_lc193_phase16_17_rescue_dagger_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_phase_records", [0] * 18)[16] != 454_656
        or dataset.get("query_outcome_success_agents") != list(SUCCESS_AGENTS)
        or dataset.get("query_outcome_failure_agents") != list(CONTROL_AGENTS)
        or control.get("target_passes") != 0
        or intervention.get("target_passes") != 256
        or rejected.get("schema") != "vq2_lc198_phase16_raw18_reroute_cem_report_v1"
        or not rejected.get("completed")
        or rejected.get("training_admitted")
        or len(rejected.get("generations", [])) != 3
        or any(item.get("target_passes") != 0 for item in rejected.get("generations", []))
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC189/LC193/LC198 do not authorize LC199")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT, LC198_REPORT,
        ROOT / "scripts/collect_vq2_lc193_phase16_17_rescue_dagger.py",
        ROOT / "scripts/train_vq2_lc136_phase8_success_only_fit.py",
        ROOT / "scripts/train_vq2_lc198_phase16_raw18_reroute_cem.py",
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


def corrected_writer(path: Path, payload: dict[str, Any]) -> None:
    corrected = dict(payload)
    if path.name == "report.json" and corrected.get("schema") == SCHEMA:
        corrected["dataset_schema"] = "vq2_lc193_phase16_17_rescue_dagger_report_v1"
        corrected["whole_puffer_phase"] = TARGET_PHASE
        corrected["fit_contract"] = "oracle-rescue success rows with separate unchanged-policy control drift constraint"
        corrected["next_authority"] = (
            "Run one exact-context LC189-versus-LC199 teacher-free raw-18 screen; no FlightSim authority."
            if corrected.get("numerically_admitted") else
            "Reject LC199 and retain LC189; do not run FlightSim."
        )
    BASE_WRITE_JSON_ONCE(path, corrected)


def configure() -> tuple[Any, ...]:
    originals = (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA, base.TARGET_PHASE,
        base.SUCCESS_AGENTS, base.CONTROL_AGENTS, base.TRAINING_SUCCESS_COUNT,
        base.OPTIMIZER_STEPS, base.LEARNING_RATE, base.SCALES,
        base.MINIMUM_SUCCESS_IMPROVEMENT, base.MAXIMUM_CONTROL_PARENT_DRIFT_MSE,
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
    base.OPTIMIZER_STEPS, base.LEARNING_RATE, base.SCALES = OPTIMIZER_STEPS, LEARNING_RATE, SCALES
    base.MINIMUM_SUCCESS_IMPROVEMENT = MINIMUM_SUCCESS_IMPROVEMENT
    base.MAXIMUM_CONTROL_PARENT_DRIFT_MSE = MAXIMUM_CONTROL_PARENT_DRIFT_MSE
    base.FROZEN_STATE_FIELD, base.FIT_METADATA_FIELD = FROZEN_STATE_FIELD, FIT_METADATA_FIELD
    base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT, PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT, base.PARENT_REPORT_SHA256 = PARENT_REPORT, PARENT_REPORT_SHA256
    base.FEATURES, base.FEATURES_SHA256 = FEATURES, FEATURES_SHA256
    base.DATASET_REPORT, base.DATASET_REPORT_SHA256 = DATASET_REPORT, DATASET_REPORT_SHA256
    base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT = PREREGISTRATION, RUNNER, TEST, DEFAULT_OUTPUT
    base.verify_inputs, base.source_identity, base.write_json_once = verify_inputs, source_identity, corrected_writer
    return originals


def restore(originals: tuple[Any, ...]) -> None:
    (
        base.TAG, base.SCHEMA, base.CHECKPOINT_SCHEMA, base.TARGET_PHASE,
        base.SUCCESS_AGENTS, base.CONTROL_AGENTS, base.TRAINING_SUCCESS_COUNT,
        base.OPTIMIZER_STEPS, base.LEARNING_RATE, base.SCALES,
        base.MINIMUM_SUCCESS_IMPROVEMENT, base.MAXIMUM_CONTROL_PARENT_DRIFT_MSE,
        base.FROZEN_STATE_FIELD, base.FIT_METADATA_FIELD,
        base.PARENT_CHECKPOINT, base.PARENT_CHECKPOINT_SHA256,
        base.PARENT_REPORT, base.PARENT_REPORT_SHA256,
        base.FEATURES, base.FEATURES_SHA256,
        base.DATASET_REPORT, base.DATASET_REPORT_SHA256,
        base.PREREGISTRATION, base.RUNNER, base.TEST, base.DEFAULT_OUTPUT,
        base.verify_inputs, base.source_identity, base.write_json_once,
    ) = originals


def fit(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
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
    return 0 if report.get("numerically_admitted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
