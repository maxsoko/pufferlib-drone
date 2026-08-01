#!/usr/bin/env python3
"""Fit all phase-6--23 Puffer residual outputs to the LC095 teacher corpus."""

from __future__ import annotations

import argparse
import json
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
from scripts.train_vq2_lc069_phase2_failure_teacher_endpoint import (
    weighted_action_mse,
    weighted_ridge,
)
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc096_late_phase_multihead_endpoint_001"
SCHEMA = "vq2_lc096_late_phase_multihead_endpoint_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc096_late_phase_multihead_endpoint_checkpoint_v1"
SEED = 431_960
PHASES = tuple(range(6, 24))
TARGET_ACTION_CLIP = 0.999
RIDGES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
VALIDATION_MODULUS = 5
VALIDATION_REMAINDER = 0
MINIMUM_PHASE_IMPROVEMENT = 1.25
MAXIMUM_PHASE_DELTA_L2 = 64.0
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc094_full_batch_serialized_checkpoint_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "14a5f90e8a7a80d3535ef0d86d3f9b5d75d42317c387a0554cf59ecf283e340a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "639edd98ec0d99c981d16837c802d758adc6622736314b3e6afe09a58af21139"
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc095_late_phase_local_teacher_features_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "8120b32a09c3e6c86d20d6c7f988b87166c249eee811f55bbd9e320a991170ff"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "fee19dfe6357362409857b493526e0dd79150f20d909c67c339ffda38d8befd8"
PREREGISTRATION = ROOT / "docs/vq2_lc096_late_phase_multihead_endpoint_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc096_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc096_late_phase_multihead_endpoint.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def split_agents(present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(present, dtype=np.int64)
    validation = values[values % VALIDATION_MODULUS == VALIDATION_REMAINDER]
    training = values[values % VALIDATION_MODULUS != VALIDATION_REMAINDER]
    if training.size == 0 or validation.size == 0:
        raise RuntimeError("LC096 phase split lacks training or validation agents")
    return training, validation


def equal_agent_weights(
    row_agents: np.ndarray, selected_agents: np.ndarray
) -> np.ndarray:
    rows = np.asarray(row_agents, dtype=np.int64)
    selected_agents = np.asarray(selected_agents, dtype=np.int64)
    selected = np.isin(rows, selected_agents)
    result = np.zeros(rows.shape, dtype=np.float64)
    counts = np.bincount(rows[selected], minlength=int(rows.max()) + 1)
    if np.any(counts[selected_agents] == 0):
        raise RuntimeError("LC096 selected an empty agent trajectory")
    result[selected] = 1.0 / (
        selected_agents.size * counts[rows[selected]]
    )
    return result


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC096 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    phases = dataset.get("feature_phase_records", [])
    if (
        parent.get("schema") != "vq2_lc094_full_batch_serialized_checkpoint_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or not parent_report.get("numerically_admitted")
        or dataset.get("schema") != "vq2_lc095_late_phase_local_teacher_features_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("failed_admission_predicates")
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_records") != FEATURES.stat().st_size // FEATURE_DTYPE.itemsize
        or len(phases) != 33
        or any(phases[phase] < 2_000 for phase in PHASES)
        or sum(phases[:6]) != 0 or sum(phases[24:]) != 0
        or dataset.get("teacher_plant_actions_executed")
        != dataset.get("total_plant_actions_executed")
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC094/LC095 do not authorize LC096")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc095_late_phase_local_teacher_features.py",
        ROOT / "scripts/train_vq2_lc069_phase2_failure_teacher_endpoint.py",
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


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    parent = verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC096 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC096 output exists without a terminal report")
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
    all_admitted = True
    for phase in PHASES:
        indices = np.flatnonzero(np.asarray(records["phase_index"]) == phase)
        phase_records = records[indices]
        row_agents = np.asarray(phase_records["agent_index"], dtype=np.int64)
        training_agents, validation_agents = split_agents(np.unique(row_agents))
        train_weights_np = equal_agent_weights(row_agents, training_agents)
        validation_weights_np = equal_agent_weights(row_agents, validation_agents)
        train_mask = train_weights_np > 0
        validation_mask = validation_weights_np > 0

        hidden = torch.from_numpy(
            np.array(phase_records["hidden"], dtype=np.float32, copy=True)
        ).to(device)
        parent_pre_tanh = torch.from_numpy(
            np.array(phase_records["base_pre_tanh"], dtype=np.float32, copy=True)
        ).to(device)
        teacher = torch.from_numpy(
            np.array(phase_records["teacher_action"], dtype=np.float32, copy=True)
        ).to(device)
        input_weight = state["indexed_phase_residual_input"][phase].to(device)
        input_bias = state["indexed_phase_residual_input_bias"][phase].to(device)
        features = torch.tanh(hidden @ input_weight.T + input_bias)
        target_delta = torch.atanh(
            teacher.clamp(-TARGET_ACTION_CLIP, TARGET_ACTION_CLIP)
        ) - parent_pre_tanh
        train_index = torch.from_numpy(np.flatnonzero(train_mask)).to(device)
        validation_index = torch.from_numpy(np.flatnonzero(validation_mask)).to(device)
        train_weights = torch.from_numpy(train_weights_np[train_mask]).to(device)
        validation_weights = torch.from_numpy(
            validation_weights_np[validation_mask]
        ).to(device)
        solutions = [
            weighted_ridge(
                features.index_select(0, train_index),
                target_delta.index_select(0, train_index),
                train_weights,
                ridge,
            )
            for ridge in RIDGES
        ]
        validation_features = features.index_select(0, validation_index)
        validation_base = parent_pre_tanh.index_select(0, validation_index)
        validation_teacher = teacher.index_select(0, validation_index)
        parent_mse = weighted_action_mse(
            torch.tanh(validation_base), validation_teacher, validation_weights
        )
        grid = []
        for ridge, solution in zip(RIDGES, solutions):
            action = torch.tanh(
                validation_base.double()
                + validation_features.double() @ solution[:-1]
                + solution[-1]
            )
            mse = weighted_action_mse(
                action, validation_teacher.double(), validation_weights
            )
            grid.append({
                "ridge": ridge,
                "teacher_action_mse": mse,
                "improvement_factor": parent_mse / max(mse, 1e-20),
                "delta_l2": float(solution.norm()),
            })
        selected_index = min(
            range(len(grid)), key=lambda index: grid[index]["teacher_action_mse"]
        )
        selected = grid[selected_index]
        solution = solutions[selected_index].detach().cpu().float()
        admitted = bool(
            selected["improvement_factor"] >= MINIMUM_PHASE_IMPROVEMENT
            and selected["delta_l2"] <= MAXIMUM_PHASE_DELTA_L2
            and torch.isfinite(solution).all()
        )
        all_admitted &= admitted
        state["indexed_phase_residual_output"][phase].add_(solution[:-1].T)
        state["indexed_phase_residual_output_bias"][phase].add_(solution[-1])
        phase_reports[str(phase)] = {
            "records": int(indices.size),
            "agents": int(np.unique(row_agents).size),
            "training_agents": int(training_agents.size),
            "validation_agents": int(validation_agents.size),
            "parent_teacher_action_mse": parent_mse,
            "grid": grid, "selected": selected, "numerically_admitted": admitted,
        }
        del hidden, parent_pre_tanh, teacher, features, target_delta, solutions

    candidate_state_hash = state_sha256(state)
    checkpoint = {
        **parent,
        "schema": CHECKPOINT_SCHEMA,
        "tag": TAG,
        "model_state": state,
        "best_epoch": 0,
        "optimizer_updates": 0,
        "numerically_admitted": bool(all_admitted),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "late_phase_endpoint": {
            "phases": list(PHASES),
            "parent_state_sha256": parent_state_hash,
            "candidate_state_sha256": candidate_state_hash,
        },
    }
    checkpoint_path = output / "policy_endpoint.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": bool(all_admitted),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_state_sha256": parent_state_hash,
        "candidate_state_sha256": candidate_state_hash,
        "feature_records": int(records.size),
        "phase_reports": phase_reports,
        "minimum_phase_improvement": MINIMUM_PHASE_IMPROVEMENT,
        "maximum_phase_delta_l2": MAXIMUM_PHASE_DELTA_L2,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "training_only_teacher_targets": True,
            "runtime_privileged_values": 0, "runtime_teacher_actions": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Run one teacher-free gate-local interpolation bracket of the complete saved Puffer endpoint."
            if all_admitted else "Reject the multi-head endpoint and retain LC094."
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
