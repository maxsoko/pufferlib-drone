#!/usr/bin/env python3
"""Fit all 1..23 long-course Puffer residual heads on LC008 features."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_public_phase import LONG_COURSE_GATE_CAP
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import (
    VQ2UnboundedProgressMLPResidualActor,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path
import scripts.train_vq2_vg068_indexed_mlp_intervention_features as base


TAG = "vq2_lc009_long_course_residual_fit_001"
SCHEMA = "vq2_lc009_long_course_residual_fit_report_v1"
STATE_SCHEMA = "vq2_lc009_long_course_residual_fit_state_v1"
CHECKPOINT_SCHEMA = "vq2_lc009_long_course_residual_checkpoint_v1"
TARGET_PHASES = tuple(range(1, 24))
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg071_nonlinear_residual_count5_multi_offset_001/policy_selected.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "60677e385cefb6d6af5c957071cb846de8396d8872880a90832d987b7494f010"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "c1494b9c56bdf2cbbfead91dc34d51d45cb780cd6317bc924835a67f7cc6b5e4"
)
DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc008_long_course_intervention_features_001"
)
DATASET_REPORT = DATASET / "report.json"
DATASET_REPORT_SHA256 = (
    "9fc1065991b4820e5500fb260ae624907628bc7d69bcf34533e5f0ec8ebfe567"
)
FEATURES = DATASET / "features.bin"
FEATURES_SHA256 = (
    "3ec8d08f1f3b9dd7aaeb0110a24aee63642232bd24cf995bc2b63378e9fb068d"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_lc009_long_course_residual_fit_preregistration_2026-07-31.md"
)
RUNNER = ROOT / "scripts/run_vq2_lc009_vast.sh"
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
CONFIG = base.TrainConfig(
    seed=431090,
    epochs=6,
    learning_rate=2e-3,
    weight_decay=1e-5,
    gradient_clip=1.0,
    residual_size=64,
    chunk_rows=65536,
    validation_agent_group_size=8,
    validation_agent_modulus=8,
    validation_agent_remainder=0,
    minimum_aggregate_improvement=2.0,
    minimum_early_phase_improvement=2.0,
    maximum_phase_regression_factor=1.0,
    maximum_trainable_l2=512.0,
)


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER,
        PARENT_CHECKPOINT, PARENT_REPORT, DATASET_REPORT, FEATURES,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_public_phase.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_lc008_long_course_intervention_features.py",
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
            raise RuntimeError(f"LC009 bound input changed: {path}")
    report = json.loads(DATASET_REPORT.read_text())
    if (
        report.get("schema") != "vq2_lc008_long_course_intervention_report_v1"
        or not report.get("completed")
        or not report.get("training_dataset_admitted")
        or report.get("failed_admission_predicates")
        or report.get("metrics", {}).get("env/success_rate") != 1.0
        or report.get("metrics", {}).get("env/crash") != 0.0
        or report.get("feature_records") != 3_631_517
        or report.get("feature_sha256") != FEATURES_SHA256
        or FEATURES.stat().st_size
        != report.get("feature_records") * report.get("feature_itemsize")
        or report.get("feature_phase_records", [])[1:24].count(0) != 0
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC008 corpus is not admitted for LC009")
    if not PREREGISTRATION.is_file() or not RUNNER.is_file():
        raise RuntimeError("LC009 source-lock surface is incomplete")
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
        parent.get("schema") != "vq2_vg071_paired_synthetic_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or contract.get("class") != "VQ2IndexedPhaseMLPResidualActor"
        or contract.get("hidden_size") != 256
        or contract.get("residual_size") != config.residual_size
    ):
        raise RuntimeError("LC009 parent actor changed")
    torch.manual_seed(config.seed)
    model = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=config.residual_size,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    model.load_converted_state(parent["model_state"])
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
    model: VQ2UnboundedProgressMLPResidualActor, parent: dict[str, Any]
) -> bool:
    reference = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=256, residual_size=64,
        initial_std=float(parent["model"]["initial_std"]),
    )
    reference.load_converted_state(parent["model_state"])
    expected = reference.state_dict()
    return all(
        torch.equal(value.detach().cpu(), expected[name].detach().cpu())
        for name, value in model.state_dict().items()
        if not name.startswith("indexed_phase_residual_")
    )


def non_target_outputs_zero(
    model: VQ2UnboundedProgressMLPResidualActor,
) -> bool:
    phases = (0, *range(24, LONG_COURSE_GATE_CAP + 1))
    return bool(all(
        torch.count_nonzero(model.indexed_phase_residual_output[phase]) == 0
        and torch.count_nonzero(
            model.indexed_phase_residual_output_bias[phase]
        ) == 0
        for phase in phases
    ))


def checkpoint_model_contract(
    parent: dict[str, Any], config: base.TrainConfig
) -> dict[str, Any]:
    return {
        **parent["model"],
        "class": "VQ2UnboundedProgressMLPResidualActor",
        "legal_observation_size": PHASE_LEGAL_OBS_SIZE,
        "legacy_legal_observation_size": LEGAL_OBS_SIZE,
        "public_phase_encoding": "active_gate_index/6 without saturation",
        "residual_heads": LONG_COURSE_GATE_CAP + 1,
        "residual_size": config.residual_size,
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
        "One source-locked teacher-free phase-local long-course screen."
    )
    base.source_paths = source_paths
    base.verify_inputs = verify_inputs
    base.source_identity = source_identity
    base.load_model = load_model
    base.phase_action_prediction = phase_action_prediction
    base.base_parameters_exact = base_parameters_exact
    base.non_target_outputs_zero = non_target_outputs_zero
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
