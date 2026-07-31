#!/usr/bin/env python3
"""Fit a phase-6 decoder endpoint on LC073-owned legal recurrent states."""

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
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
from scripts.train_vq2_lc069_phase2_failure_teacher_endpoint import (
    weighted_action_mse,
    weighted_ridge,
)
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc076_phase6_student_decoder_endpoint_001"
SCHEMA = "vq2_lc076_phase6_student_decoder_endpoint_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc076_phase6_student_decoder_endpoint_checkpoint_v1"
SEED = 431760
TARGET_PHASE = 6
TARGET_ACTION_CLIP = 0.999
RIDGES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
MINIMUM_IMPROVEMENT = 1.10
MAXIMUM_ENDPOINT_L2 = 64.0
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc073_phase2_constrained_endpoint_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "5614a95cd2f9b428a99ddd43ec5e324c095344cae51b8d771aea3d5f05d35be8"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "ae68af50524b02425f9a970c7f87dccbcbb6b60a0debb9899fcf1baa167b5d75"
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc075_phase6_horizon_capped_features_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "092143c6f27677f77e718fb9a0f7e7c65580485f52d5226d529d798212e1ac83"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "0be7acea06bff0ab46591c67b1f78125103d82672018e5e320bfd30cb9836d76"
PREREGISTRATION = ROOT / "docs/vq2_lc076_phase6_student_decoder_endpoint_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc076_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc076_phase6_student_decoder_endpoint.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def agent_split(query_agents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(query_agents, dtype=np.int64).copy()
    rng = np.random.default_rng(SEED)
    rng.shuffle(values)
    validation_count = max(1, int(round(0.2 * values.size)))
    return np.sort(values[validation_count:]), np.sort(values[:validation_count])


def equal_trajectory_weights(
    agents: np.ndarray, selected_agents: np.ndarray
) -> np.ndarray:
    agents = np.asarray(agents, dtype=np.int64)
    selected_agents = np.asarray(selected_agents, dtype=np.int64)
    selected = np.isin(agents, selected_agents)
    result = np.zeros(agents.shape, dtype=np.float64)
    counts = np.bincount(agents[selected], minlength=int(agents.max()) + 1)
    if selected_agents.size == 0 or np.any(counts[selected_agents] == 0):
        raise ValueError("LC076 split contains an empty trajectory")
    result[selected] = 1.0 / (
        selected_agents.size * counts[agents[selected]]
    )
    if not np.isclose(result.sum(), 1.0, atol=1e-10):
        raise RuntimeError("LC076 trajectory weights do not sum to one")
    return result


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC076 bound input changed: {path}")
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    phases = dataset.get("feature_phase_records", [])
    if (
        parent_report.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or parent.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or dataset.get("schema") != "vq2_lc075_phase6_horizon_capped_features_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("failed_admission_predicates")
        or dataset.get("feature_records") != 20_007
        or dataset.get("query_agents") != 18
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or len(phases) != 33 or phases[TARGET_PHASE] != 20_007
        or sum(value for index, value in enumerate(phases) if index != TARGET_PHASE) != 0
        or dataset.get("teacher_plant_actions_executed") != 0
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC073/LC075 do not authorize LC076")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc075_phase6_horizon_capped_features.py",
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
        raise RuntimeError("LC076 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC076 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    device = torch.device(device_name)
    started = time.perf_counter()

    records = np.memmap(FEATURES, dtype=FEATURE_DTYPE, mode="r")
    agents = np.asarray(records["agent_index"], dtype=np.int64)
    query_agents = np.unique(agents)
    if query_agents.size != 18:
        raise RuntimeError("LC076 source query-agent count changed")
    train_agents, validation_agents = agent_split(query_agents)
    train_weights_np = equal_trajectory_weights(agents, train_agents)
    validation_weights_np = equal_trajectory_weights(agents, validation_agents)
    train_mask = train_weights_np > 0
    validation_mask = validation_weights_np > 0

    state = parent["model_state"]
    hidden = torch.from_numpy(
        np.array(records["hidden"], dtype=np.float32, copy=True)
    ).to(device)
    input_weight = state["indexed_phase_residual_input"][TARGET_PHASE].to(device)
    input_bias = state["indexed_phase_residual_input_bias"][TARGET_PHASE].to(device)
    parent_weight = state["indexed_phase_residual_output"][TARGET_PHASE].to(device)
    parent_bias = state["indexed_phase_residual_output_bias"][TARGET_PHASE].to(device)
    action_weight = state["action_head.weight"].to(device)
    action_bias = state["action_head.bias"].to(device)
    features = torch.tanh(hidden @ input_weight.T + input_bias)
    base_pre_tanh = hidden @ action_weight.T + action_bias
    teacher = torch.from_numpy(
        np.array(records["teacher_action"], dtype=np.float32, copy=True)
    ).to(device)
    target = torch.atanh(
        teacher.clamp(-TARGET_ACTION_CLIP, TARGET_ACTION_CLIP)
    ) - base_pre_tanh
    train_index = torch.from_numpy(np.flatnonzero(train_mask)).to(device)
    validation_index = torch.from_numpy(np.flatnonzero(validation_mask)).to(device)
    train_weights = torch.from_numpy(train_weights_np[train_mask]).to(device)
    validation_weights = torch.from_numpy(validation_weights_np[validation_mask]).to(device)
    solutions = [
        weighted_ridge(
            features.index_select(0, train_index),
            target.index_select(0, train_index),
            train_weights,
            ridge,
        )
        for ridge in RIDGES
    ]
    validation_features = features.index_select(0, validation_index)
    validation_base = base_pre_tanh.index_select(0, validation_index)
    validation_teacher = teacher.index_select(0, validation_index)
    parent_action = torch.tanh(
        validation_base + validation_features @ parent_weight.T + parent_bias
    )
    parent_mse = weighted_action_mse(
        parent_action, validation_teacher, validation_weights
    )
    grid: list[dict[str, float]] = []
    for ridge, solution in zip(RIDGES, solutions):
        candidate = torch.tanh(
            validation_base.double()
            + validation_features.double() @ solution[:-1] + solution[-1]
        )
        mse = weighted_action_mse(
            candidate, validation_teacher.double(), validation_weights
        )
        grid.append({
            "ridge": ridge, "teacher_action_mse": mse,
            "improvement_factor": parent_mse / max(mse, 1e-20),
            "endpoint_l2": float(solution.norm()),
        })
    selected_index = min(range(len(grid)), key=lambda index: grid[index]["teacher_action_mse"])
    selected = grid[selected_index]
    solution = solutions[selected_index].detach().cpu().float()
    output_weight = solution[:-1].T.contiguous()
    output_bias = solution[-1].contiguous()
    admitted = bool(
        selected["improvement_factor"] >= MINIMUM_IMPROVEMENT
        and selected["endpoint_l2"] <= MAXIMUM_ENDPOINT_L2
        and torch.isfinite(output_weight).all() and torch.isfinite(output_bias).all()
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "numerically_admitted": admitted,
        "target_phase": TARGET_PHASE,
        "output_weight": output_weight, "output_bias": output_bias,
        "parent_output_weight": parent_weight.detach().cpu(),
        "parent_output_bias": parent_bias.detach().cpu(),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "selected_ridge": selected["ridge"], "validation": selected,
    }
    checkpoint_path = output / "decoder_endpoint.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "wall_time_seconds": time.perf_counter() - started,
        "feature_records": int(records.size), "query_agents": int(query_agents.size),
        "train_agents": int(train_agents.size),
        "validation_agents": int(validation_agents.size),
        "parent_teacher_action_mse": parent_mse,
        "grid": grid, "selected": selected,
        "minimum_improvement_factor": MINIMUM_IMPROVEMENT,
        "maximum_endpoint_l2": MAXIMUM_ENDPOINT_L2,
        "source_identity": identity,
        "safety": {
            "training_only_teacher_targets": True,
            "runtime_privileged_values": 0, "runtime_teacher_actions": 0,
            "student_updates": 0, "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Run one fast teacher-free phase-6 decoder interpolation milestone screen."
            if admitted else "Reject the phase-6 endpoint and retain LC073."
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
