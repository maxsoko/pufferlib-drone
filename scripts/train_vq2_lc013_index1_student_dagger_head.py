#!/usr/bin/env python3
"""Fit only LC010S public-index-1 head on LC012 student-state labels."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_public_phase import LONG_COURSE_GATE_CAP
from pufferlib.vq2_recurrent_phase_residual import (
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_vg068_indexed_mlp_intervention_features as base


TAG = "vq2_lc013_index1_student_dagger_head_001"
SCHEMA = "vq2_lc013_index1_student_dagger_head_report_v1"
STATE_SCHEMA = "vq2_lc013_index1_student_dagger_head_state_v1"
CHECKPOINT_SCHEMA = "vq2_lc013_index1_student_dagger_head_checkpoint_v1"
TARGET_PHASES = (1,)
PARENT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc010s_phase_independent_early_stop_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "34747327c2f431a6153d200891bad45b8431a5a5d0fe2418108ae822aa716d4d"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "afae8fa795159824bd447472b7eb5047167f19a4848264c95593c08ab43af21f"
)
DATASET = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc012_index1_student_dagger_features_001"
)
DATASET_REPORT = DATASET / "report.json"
DATASET_REPORT_SHA256 = (
    "7db385a18370953ab6b624e64cec0f84bfa7f8af4cd0699c2bc82c3c374c75ac"
)
FEATURES = DATASET / "features.bin"
FEATURES_SHA256 = (
    "6d0402bd40d7dc4202e8ef23dceebbc35f7acb75f5d558673cd126fdde6fd049"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc013_index1_student_dagger_head_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc013_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc013_index1_student_dagger_head.py"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CONFIG = base.TrainConfig(
    seed=431130,
    epochs=10,
    learning_rate=2e-4,
    weight_decay=0.0,
    gradient_clip=1.0,
    residual_size=64,
    chunk_rows=32768,
    validation_agent_group_size=8,
    validation_agent_modulus=8,
    validation_agent_remainder=0,
    minimum_aggregate_improvement=1.02,
    minimum_early_phase_improvement=1.02,
    maximum_phase_regression_factor=1.0,
    maximum_trainable_l2=512.0,
)
RESIDUAL_NAMES = (
    "indexed_phase_residual_input",
    "indexed_phase_residual_input_bias",
    "indexed_phase_residual_output",
    "indexed_phase_residual_output_bias",
)


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, DATASET_REPORT, FEATURES,
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_lc012_index1_student_dagger_features.py",
        ROOT / "scripts/train_vq2_vg068_indexed_mlp_intervention_features.py",
    )


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC013 bound input changed: {path}")
    parent = json.loads(PARENT_REPORT.read_text())
    report = json.loads(DATASET_REPORT.read_text())
    phases = report.get("feature_phase_records", [])
    if (
        parent.get("schema")
        != "vq2_lc010s_phase_independent_early_stop_report_v1"
        or not parent.get("numerically_admitted")
        or parent.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or report.get("schema") != "vq2_lc012_index1_student_dagger_report_v1"
        or not report.get("training_dataset_admitted")
        or report.get("failed_admission_predicates")
        or report.get("feature_records") != 215_813
        or report.get("feature_sha256") != FEATURES_SHA256
        or FEATURES.stat().st_size
        != report.get("feature_records") * report.get("feature_itemsize")
        or len(phases) != LONG_COURSE_GATE_CAP + 1
        or phases[1] != report.get("feature_records")
        or sum(phases[:1] + phases[2:]) != 0
        or report.get("teacher_plant_actions_executed") != 0
        or report.get("student_plant_actions_executed")
        != report.get("total_plant_actions_executed")
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC012 is not the admitted phase-1 DAgger corpus")
    return report


def source_identity() -> tuple[str, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return commit, {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths()
    }


def load_model(
    device: torch.device, config: base.TrainConfig = CONFIG,
) -> tuple[VQ2UnboundedProgressMLPResidualActor, dict[str, Any]]:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = parent.get("model", {})
    if (
        parent.get("schema")
        != "vq2_lc010s_phase_independent_early_stop_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or contract.get("class") != "VQ2UnboundedProgressMLPResidualActor"
        or contract.get("hidden_size") != 256
        or contract.get("residual_size") != config.residual_size
        or contract.get("residual_heads") != LONG_COURSE_GATE_CAP + 1
    ):
        raise RuntimeError("LC013 parent Puffer contract changed")
    model = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=config.residual_size,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    model.load_state_dict(parent["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in base.trainable_parameters(model):
        parameter.requires_grad_(True)
    return model, parent


def phase_action_prediction(
    model: VQ2UnboundedProgressMLPResidualActor,
    hidden: torch.Tensor,
    stored_parent_pre_tanh: torch.Tensor,
    phase: int,
) -> torch.Tensor:
    del stored_parent_pre_tanh
    trunk = torch.nn.functional.linear(
        hidden, model.action_head.weight, model.action_head.bias
    )
    feature = torch.tanh(
        hidden @ model.indexed_phase_residual_input[phase].T
        + model.indexed_phase_residual_input_bias[phase]
    )
    residual = (
        feature @ model.indexed_phase_residual_output[phase].T
        + model.indexed_phase_residual_output_bias[phase]
    )
    return torch.tanh(trunk + residual)


def base_parameters_exact(
    model: VQ2UnboundedProgressMLPResidualActor, parent: dict[str, Any],
) -> bool:
    return all(
        torch.equal(value.detach().cpu(), parent["model_state"][name])
        for name, value in model.state_dict().items()
        if name not in RESIDUAL_NAMES
    )


def non_target_outputs_zero(
    model: VQ2UnboundedProgressMLPResidualActor,
) -> bool:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    keep = torch.arange(LONG_COURSE_GATE_CAP + 1) != TARGET_PHASES[0]
    return all(
        torch.equal(
            getattr(model, name).detach().cpu()[keep],
            parent["model_state"][name][keep],
        )
        for name in RESIDUAL_NAMES
    )


def numerically_admitted(
    metrics: dict[str, Any], *, base_exact: bool, non_target_zero: bool,
    trainable_l2: float, config: base.TrainConfig = CONFIG,
) -> bool:
    phase = metrics["phases"][str(TARGET_PHASES[0])]
    return bool(
        base_exact
        and non_target_zero
        and math.isfinite(trainable_l2)
        and trainable_l2 <= config.maximum_trainable_l2
        and phase["selected_action_mse"] <= phase["baseline_action_mse"]
        and phase["improvement_factor"] >= config.minimum_aggregate_improvement
    )


def checkpoint_model_contract(
    parent: dict[str, Any], config: base.TrainConfig,
) -> dict[str, Any]:
    return {
        **parent["model"],
        "class": "VQ2UnboundedProgressMLPResidualActor",
        "residual_size": config.residual_size,
        "residual_heads": LONG_COURSE_GATE_CAP + 1,
    }


def configure() -> None:
    base.TAG = TAG
    base.SCHEMA = SCHEMA
    base.STATE_SCHEMA = STATE_SCHEMA
    base.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    base.TARGET_PHASES = TARGET_PHASES
    base.PARENT_CHECKPOINT = PARENT_CHECKPOINT
    base.PARENT_CHECKPOINT_SHA256 = PARENT_CHECKPOINT_SHA256
    base.PARENT_REPORT = PARENT_REPORT
    base.PARENT_REPORT_SHA256 = PARENT_REPORT_SHA256
    base.DATASET = DATASET
    base.DATASET_REPORT = DATASET_REPORT
    base.DATASET_REPORT_SHA256 = DATASET_REPORT_SHA256
    base.FEATURES = FEATURES
    base.FEATURES_SHA256 = FEATURES_SHA256
    base.DATASET_ADMISSION = DATASET_REPORT
    base.DATASET_ADMISSION_SHA256 = DATASET_REPORT_SHA256
    base.PREREGISTRATION = PREREGISTRATION
    base.RUNNER = RUNNER
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.CONFIG = CONFIG
    base.MODEL_CLASS_NAME = "VQ2UnboundedProgressMLPResidualActor"
    base.NEXT_AUTHORITY_ADMITTED = (
        "One source-locked bounded teacher-free LC013 prefix screen."
    )
    base.source_paths = source_paths
    base.verify_inputs = verify_inputs
    base.source_identity = source_identity
    base.load_model = load_model
    base.phase_action_prediction = phase_action_prediction
    base.base_parameters_exact = base_parameters_exact
    base.non_target_outputs_zero = non_target_outputs_zero
    base.numerically_admitted = numerically_admitted
    base.checkpoint_model_contract = checkpoint_model_contract


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    configure()
    return base.fit(
        output=output, device_name=device_name, resume=resume, config=CONFIG
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = fit(
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
