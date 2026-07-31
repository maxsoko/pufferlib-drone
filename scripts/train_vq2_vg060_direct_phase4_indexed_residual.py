#!/usr/bin/env python3
"""Fit only indexed residual head 4 on warmed VG039 phase-4 rows."""

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

from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseResidualActor
from scripts.train_vq2_variable_gate_recurrent_bc import sha256_path
import scripts.train_vq2_vg057_warmed_phase_residual as core


TAG = "vq2_vg060_direct_phase4_indexed_residual_001"
REPORT_SCHEMA = "vq2_vg060_direct_phase4_indexed_residual_report_v1"
CHECKPOINT_SCHEMA = "vq2_vg060_direct_phase4_indexed_residual_checkpoint_v1"
STATE_SCHEMA = "vq2_vg060_direct_phase4_indexed_residual_state_v1"
TARGET_PHASE_INDEX = 4
PHASE_ROW_WEIGHTS = tuple(1.0 if index == TARGET_PHASE_INDEX else 0.0 for index in range(17))
REJECTION = ROOT / "docs/vq2_vg059_phase4_indexed_residual_bracket_rejection_2026-07-31.json"
REJECTION_SHA256 = "a181066247c73ab7222323a945a0c777c6ba72f3d25ab2992e0255f7755e858b"
PREREGISTRATION = ROOT / "docs/vq2_vg060_direct_phase4_indexed_residual_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg060_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
CONFIG = core.TrainConfig(
    seed=429192,
    epochs=12,
    learning_rate=1e-3,
    phase_row_weights=PHASE_ROW_WEIGHTS,
)


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, core.GOAL,
        core.PARENT_CHECKPOINT, core.PARENT_REPORT, core.PARENT_ADMISSION,
        core.DATASET / "report.json", core.DATASET / "metadata.json",
        core.DATASET_ADMISSION, REJECTION,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        ROOT / "scripts/train_vq2_vg057_warmed_phase_residual.py",
        ROOT / "scripts/train_vq2_variable_gate_recurrent_bc.py",
    )


def source_identity() -> tuple[str, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return commit, {
        str(path.resolve().relative_to(ROOT)): sha256_path(path)
        for path in source_paths()
    }


def verify_inputs() -> None:
    expected = {
        core.PARENT_CHECKPOINT: core.PARENT_CHECKPOINT_SHA256,
        core.PARENT_REPORT: core.PARENT_REPORT_SHA256,
        core.PARENT_ADMISSION: core.PARENT_ADMISSION_SHA256,
        core.DATASET / "report.json": core.DATASET_REPORT_SHA256,
        core.DATASET / "metadata.json": core.DATASET_METADATA_SHA256,
        core.DATASET_ADMISSION: core.DATASET_ADMISSION_SHA256,
        REJECTION: REJECTION_SHA256,
        core.GOAL: core.GOAL_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG060 source evidence changed: {path}")
    if not PREREGISTRATION.is_file() or not RUNNER.is_file():
        raise RuntimeError("VG060 source lock is incomplete")
    rejection = json.loads(REJECTION.read_text())
    dataset = json.loads((core.DATASET / "report.json").read_text())
    if (
        rejection.get("schema") != "vq2_vg059_phase4_indexed_residual_bracket_rejection_v1"
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("live_authority")
    ):
        raise RuntimeError("VG059 rejection does not authorize VG060")
    if (
        not dataset.get("admitted")
        or not dataset.get("admission_predicates", {}).get("minimum_phase_4_records")
        or dataset.get("safety", {}).get("teacher_actions_executed") != 0
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("VG039 warmed phase-4 source changed")


def load_model(device: torch.device) -> tuple[VQ2IndexedPhaseResidualActor, dict[str, Any]]:
    payload = torch.load(core.PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or contract.get("class") != "VQ2PhaseRecurrentActor"
        or contract.get("legal_observation_size") != PHASE_LEGAL_OBS_SIZE
    ):
        raise RuntimeError("VG060 parent actor changed")
    model = VQ2IndexedPhaseResidualActor(hidden_size=256, initial_std=0.15).to(device)
    model.load_base_state(payload["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.indexed_phase_action_residual.requires_grad_(True)
    return model, payload


def base_exact(model: VQ2IndexedPhaseResidualActor, parent: dict[str, Any]) -> bool:
    state = model.state_dict()
    if not all(
        torch.equal(state[name].cpu(), value.cpu())
        for name, value in parent["model_state"].items()
    ):
        return False
    residual = state["indexed_phase_action_residual"]
    return bool(
        torch.count_nonzero(residual[:TARGET_PHASE_INDEX]) == 0
        and torch.count_nonzero(residual[TARGET_PHASE_INDEX + 1 :]) == 0
    )


def trainable_parameters(model: VQ2IndexedPhaseResidualActor) -> list[torch.nn.Parameter]:
    return [model.indexed_phase_action_residual]


def trained_parameter_norm(model: VQ2IndexedPhaseResidualActor) -> float:
    return float(
        model.indexed_phase_action_residual[TARGET_PHASE_INDEX].detach().norm().item()
    )


def numerically_admitted(
    baseline: dict[str, Any], selected: dict[str, Any], *,
    base_parameters_exact: bool, residual_norm: float,
) -> bool:
    b, s = baseline["phase_weighted_mse"], selected["phase_weighted_mse"]
    return bool(
        base_parameters_exact
        and selected["phase_counts"].get("4", 0) >= 1000
        and s["4"] < b["4"]
        and all(s[str(index)] == b[str(index)] for index in range(4))
        and np.isfinite(residual_norm)
        and residual_norm <= 16.0
    )


def configure() -> None:
    core.TAG = TAG; core.REPORT_SCHEMA = REPORT_SCHEMA
    core.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA; core.STATE_SCHEMA = STATE_SCHEMA
    core.PHASE_ROW_WEIGHTS = PHASE_ROW_WEIGHTS
    core.REJECTION = REJECTION; core.REJECTION_SHA256 = REJECTION_SHA256
    core.PREREGISTRATION = PREREGISTRATION; core.RUNNER = RUNNER
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.MODEL_CLASS_NAME = "VQ2IndexedPhaseResidualActor"
    core.TRAINABLE_PARAMETER_NAMES = ("indexed_phase_action_residual[4]",)
    core.TRAINABLE_PARAMETER_COUNT = 17 * 4 * 256
    core.verify_inputs = verify_inputs; core.source_paths = source_paths
    core.source_identity = source_identity; core.load_model = load_model
    core.base_exact = base_exact; core.trainable_parameters = trainable_parameters
    core.trained_parameter_norm = trained_parameter_norm
    core.numerically_admitted = numerically_admitted


def train(*, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False) -> dict[str, Any]:
    configure()
    return core.train(output=output, device_name=device_name, config=CONFIG, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
