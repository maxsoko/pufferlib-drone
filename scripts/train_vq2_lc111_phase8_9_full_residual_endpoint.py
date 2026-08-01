#!/usr/bin/env python3
"""Fit complete phase-8/9 residual MLP heads on the dense LC095 corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_vq2_vg062_warmed_teacher_intervention_features import FEATURE_DTYPE
from scripts.eval_vq2_lc015_restore_vg071_head1 import state_sha256
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.train_vq2_lc069_phase2_failure_teacher_endpoint import weighted_action_mse
from scripts.train_vq2_lc096_late_phase_multihead_endpoint import (
    equal_agent_weights,
    split_agents,
)
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc111_phase8_9_full_residual_endpoint_001"
SCHEMA = "vq2_lc111_phase8_9_full_residual_endpoint_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc111_phase8_9_full_residual_endpoint_checkpoint_v1"
SEED = 432_110
PHASES = (8, 9)
OPTIMIZER_STEPS = 512
BATCH_SIZE = 4_096
EVALUATION_INTERVAL = 16
LEARNING_RATE = 3e-3
ANCHOR_COEFFICIENT = 1e-3
GRADIENT_CLIP = 1.0
TARGET_ACTION_CLIP = 0.999
MINIMUM_PHASE_IMPROVEMENT = 1.50
MAXIMUM_PHASE_DELTA_L2 = 64.0
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc105_phase7_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "005e5e7929258fd282ab390fd230fda817700afaee790e78144200e67b3e10a4"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "c614e6929282399cac2f18465084f8337ed8770389a74e48b86ff1fcf4315e7d"
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc095_late_phase_local_teacher_features_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "8120b32a09c3e6c86d20d6c7f988b87166c249eee811f55bbd9e320a991170ff"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "fee19dfe6357362409857b493526e0dd79150f20d909c67c339ffda38d8befd8"
LC110_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc110_phase8_success_action_bias_001/report.json"
)
LC110_REPORT_SHA256 = "0310afe43e4cd80d9b880fc57d0d31df9b222a708449636827b37440bf5c4d39"
PREREGISTRATION = ROOT / "docs/vq2_lc111_phase8_9_full_residual_endpoint_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc111_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc111_phase8_9_full_residual_endpoint.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PARAMETER_NAMES = (
    "indexed_phase_residual_input",
    "indexed_phase_residual_input_bias",
    "indexed_phase_residual_output",
    "indexed_phase_residual_output_bias",
)


def residual_action(
    hidden: torch.Tensor,
    base: torch.Tensor,
    parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    input_weight, input_bias, output_weight, output_bias = parameters
    feature = torch.tanh(hidden @ input_weight.T + input_bias)
    return torch.tanh(base + feature @ output_weight.T + output_bias)


def parameter_delta_l2(
    parameters: tuple[torch.Tensor, ...], parents: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return torch.sqrt(sum(
        (value - parent).double().square().sum()
        for value, parent in zip(parameters, parents)
    ))


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        LC110_REPORT: LC110_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC111 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC110_REPORT.read_text())
    phase_records = dataset.get("feature_phase_records", [])
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema") != "vq2_lc095_late_phase_local_teacher_features_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or len(phase_records) != 33
        or phase_records[8] != 62_896
        or phase_records[9] != 58_288
        or dataset.get("teacher_plant_actions_executed")
        != dataset.get("total_plant_actions_executed")
        or rejected.get("schema") != "vq2_lc110_phase8_success_action_bias_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or any(item.get("target_passes") != 0 for item in rejected.get("items", [])[1:])
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC095/LC105/LC110 do not authorize LC111")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        LC110_REPORT,
        ROOT / "scripts/collect_vq2_lc095_late_phase_local_teacher_features.py",
        ROOT / "scripts/train_vq2_lc096_late_phase_multihead_endpoint.py",
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


def fit_phase(
    phase_records: np.ndarray,
    state: dict[str, torch.Tensor],
    phase: int,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    row_agents = np.asarray(phase_records["agent_index"], dtype=np.int64)
    training_agents, validation_agents = split_agents(np.unique(row_agents))
    train_weights_np = equal_agent_weights(row_agents, training_agents)
    validation_weights_np = equal_agent_weights(row_agents, validation_agents)
    train_index = np.flatnonzero(train_weights_np > 0)
    validation_index = np.flatnonzero(validation_weights_np > 0)

    hidden = torch.from_numpy(
        np.array(phase_records["hidden"], dtype=np.float32, copy=True)
    ).to(device)
    teacher = torch.from_numpy(
        np.array(phase_records["teacher_action"], dtype=np.float32, copy=True)
    ).to(device).clamp(-TARGET_ACTION_CLIP, TARGET_ACTION_CLIP)
    action_weight = state["action_head.weight"].to(device)
    action_bias = state["action_head.bias"].to(device)
    base = hidden @ action_weight.T + action_bias
    parents = tuple(
        state[name][phase].to(device).detach().clone() for name in PARAMETER_NAMES
    )
    parameters = tuple(value.clone().requires_grad_(True) for value in parents)
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    train_index_tensor = torch.from_numpy(train_index).to(device)
    train_probabilities = torch.from_numpy(
        train_weights_np[train_index].astype(np.float32)
    ).to(device)
    train_probabilities /= train_probabilities.sum()
    validation_index_tensor = torch.from_numpy(validation_index).to(device)
    validation_weights = torch.from_numpy(
        validation_weights_np[validation_index].astype(np.float64)
    ).to(device)
    with torch.no_grad():
        parent_validation_action = residual_action(
            hidden.index_select(0, validation_index_tensor),
            base.index_select(0, validation_index_tensor),
            parents,
        )
        parent_mse = weighted_action_mse(
            parent_validation_action,
            teacher.index_select(0, validation_index_tensor),
            validation_weights,
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + phase)
    best_step = 0
    best_mse = parent_mse
    best_parameters = tuple(value.detach().clone() for value in parents)
    history: list[dict[str, float | int]] = []
    for step in range(1, OPTIMIZER_STEPS + 1):
        sampled = torch.multinomial(
            train_probabilities, BATCH_SIZE, replacement=True, generator=generator
        )
        indices = train_index_tensor.index_select(0, sampled)
        prediction = residual_action(
            hidden.index_select(0, indices),
            base.index_select(0, indices),
            parameters,
        )
        action_loss = torch.mean(
            (prediction - teacher.index_select(0, indices)).square()
        )
        anchor = sum(
            (value - parent).square().mean()
            for value, parent in zip(parameters, parents)
        )
        loss = action_loss + ANCHOR_COEFFICIENT * anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
        optimizer.step()
        if step % EVALUATION_INTERVAL == 0 or step == OPTIMIZER_STEPS:
            with torch.no_grad():
                validation_action = residual_action(
                    hidden.index_select(0, validation_index_tensor),
                    base.index_select(0, validation_index_tensor),
                    parameters,
                )
                validation_mse = weighted_action_mse(
                    validation_action,
                    teacher.index_select(0, validation_index_tensor),
                    validation_weights,
                )
            history.append({
                "step": step,
                "training_action_mse": float(action_loss.detach()),
                "validation_teacher_action_mse": validation_mse,
            })
            if math.isfinite(validation_mse) and validation_mse < best_mse:
                best_step = step
                best_mse = validation_mse
                best_parameters = tuple(
                    value.detach().clone() for value in parameters
                )

    delta_l2 = float(parameter_delta_l2(best_parameters, parents))
    improvement = parent_mse / max(best_mse, 1e-20)
    admitted = bool(
        best_step > 0
        and improvement >= MINIMUM_PHASE_IMPROVEMENT
        and delta_l2 <= MAXIMUM_PHASE_DELTA_L2
        and all(bool(torch.isfinite(value).all()) for value in best_parameters)
    )
    report = {
        "records": int(phase_records.size),
        "agents": int(np.unique(row_agents).size),
        "training_agents": int(training_agents.size),
        "validation_agents": int(validation_agents.size),
        "parent_teacher_action_mse": parent_mse,
        "best_validation_teacher_action_mse": best_mse,
        "improvement_factor": improvement,
        "parameter_delta_l2": delta_l2,
        "best_step": best_step,
        "history": history,
        "numerically_admitted": admitted,
    }
    return report, tuple(value.detach().cpu().float() for value in best_parameters)


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    parent = verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC111 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC111 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    started = time.perf_counter()
    device = torch.device(device_name)
    records = np.memmap(FEATURES, dtype=FEATURE_DTYPE, mode="r")
    state = {
        name: value.detach().cpu().clone()
        for name, value in parent["model_state"].items()
    }
    parent_state_hash = state_sha256(state)
    phase_reports: dict[str, Any] = {}
    for phase in PHASES:
        indices = np.flatnonzero(np.asarray(records["phase_index"]) == phase)
        phase_report, parameters = fit_phase(
            records[indices], state, phase, device=device
        )
        phase_reports[str(phase)] = phase_report
        for name, value in zip(PARAMETER_NAMES, parameters):
            state[name][phase].copy_(value)
    all_admitted = all(
        report["numerically_admitted"] for report in phase_reports.values()
    )
    candidate_state_hash = state_sha256(state)
    checkpoint = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": TAG,
        "model_state": state,
        "best_epoch": 0,
        "optimizer_updates": OPTIMIZER_STEPS * len(PHASES),
        "numerically_admitted": all_admitted,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "full_residual_endpoint": {
            "phases": list(PHASES),
            "parent_state_sha256": parent_state_hash,
            "candidate_state_sha256": candidate_state_hash,
        },
    }
    checkpoint_path = output / "policy_endpoint.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": all_admitted,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_state_sha256": parent_state_hash,
        "candidate_state_sha256": candidate_state_hash,
        "phases": list(PHASES), "optimizer_steps_per_phase": OPTIMIZER_STEPS,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "anchor_coefficient": ANCHOR_COEFFICIENT,
        "minimum_phase_improvement": MINIMUM_PHASE_IMPROVEMENT,
        "maximum_phase_delta_l2": MAXIMUM_PHASE_DELTA_L2,
        "phase_reports": phase_reports,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "training_only_teacher_targets": True,
            "runtime_privileged_values": 0, "runtime_teacher_actions": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Run one teacher-free pairwise-256 scale bracket of the complete saved-form endpoint."
            if all_admitted else "Reject the full-residual endpoint and retain LC105."
        ),
    }
    write_json_once(report_path, report)
    return report


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
