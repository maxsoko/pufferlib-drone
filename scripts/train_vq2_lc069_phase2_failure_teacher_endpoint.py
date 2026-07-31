#!/usr/bin/env python3
"""Fit a success-anchored, failure-teacher phase-2 Puffer decoder endpoint."""

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
from scripts.train_vq2_lc065_phase2_failure_direction import terminal_outcome_by_agent
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc069_phase2_failure_teacher_endpoint_001"
SCHEMA = "vq2_lc069_phase2_failure_teacher_endpoint_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc069_phase2_failure_teacher_endpoint_checkpoint_v1"
SEED = 431690
TARGET_PHASE = 2
FEATURE_CHUNK = 65_536
TARGET_ACTION_CLIP = 0.999
RIDGES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
MINIMUM_FAILURE_IMPROVEMENT = 1.10
MAXIMUM_SUCCESS_ACTION_DRIFT_MSE = 0.0025
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc062_phase2_bias_full_course_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "06381161445f5f207a3b12f7c97b82c884f91a71fdb3e34e05189255cc58d04a"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "5bf0fc1f95fec795dcf0c8212f7838a0f18a683baaece679228e87e667085af5"
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc028_index2_student_dagger_features_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "4a74f8178a7f3abd33bba8f1ec2009245c53b5dcb06bf03b26267f96f7650701"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "ec8d760edeca4f07c4238f6edd76fb4db7ebc955bf0506f619020615b0ba1a1e"
LC068_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc068_phase2_failure_full_course_001/report.json"
)
LC068_REPORT_SHA256 = "5857793ff2927ce023dd4c5dd77f9f37057a0c9e1c663899c7e7c780b9abace5"
PREREGISTRATION = ROOT / "docs/vq2_lc069_phase2_failure_teacher_endpoint_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc069_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc069_phase2_failure_teacher_endpoint.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def stratified_agent_split(
    present: np.ndarray, outcome: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    train: list[int] = []
    validation: list[int] = []
    for label in (0, 1):
        ids = present[outcome[present] == label].copy()
        rng.shuffle(ids)
        cut = max(1, min(ids.size - 1, int(round(0.8 * ids.size))))
        train.extend(ids[:cut].tolist())
        validation.extend(ids[cut:].tolist())
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(validation), dtype=np.int64)


def trajectory_class_weights(
    agents: np.ndarray, outcome: np.ndarray, selected_agents: np.ndarray
) -> np.ndarray:
    agents = np.asarray(agents, dtype=np.int64)
    selected_agents = np.asarray(selected_agents, dtype=np.int64)
    selected = np.isin(agents, selected_agents)
    result = np.zeros(agents.shape, dtype=np.float64)
    counts = np.bincount(agents[selected], minlength=outcome.size)
    for label in (0, 1):
        class_agents = selected_agents[outcome[selected_agents] == label]
        if class_agents.size == 0:
            raise ValueError("LC069 split lacks an outcome class")
        class_mask = selected & np.isin(agents, class_agents)
        result[class_mask] = 0.5 / (
            class_agents.size * counts[agents[class_mask]]
        )
    if not np.isclose(result.sum(), 1.0, atol=1e-10):
        raise RuntimeError("LC069 trajectory/class weights do not sum to one")
    return result


def asymmetric_residual_target(
    base_pre_tanh: torch.Tensor,
    parent_residual: torch.Tensor,
    teacher_action: torch.Tensor,
    failure: torch.Tensor,
) -> torch.Tensor:
    teacher_residual = torch.atanh(
        teacher_action.clamp(-TARGET_ACTION_CLIP, TARGET_ACTION_CLIP)
    ) - base_pre_tanh
    return torch.where(failure[:, None], teacher_residual, parent_residual)


def weighted_ridge(
    features: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, ridge: float
) -> torch.Tensor:
    if features.ndim != 2 or target.ndim != 2 or features.shape[0] != target.shape[0]:
        raise ValueError("LC069 ridge rows do not align")
    if weights.shape != (features.shape[0],):
        raise ValueError("LC069 ridge weights do not align")
    x = torch.cat(
        (features.double(), torch.ones(features.shape[0], 1, dtype=torch.float64, device=features.device)),
        dim=1,
    )
    y = target.double()
    normalized = weights.double() / weights.double().sum()
    gram = x.T @ (x * normalized[:, None])
    cross = x.T @ (y * normalized[:, None])
    regularizer = torch.eye(x.shape[1], dtype=torch.float64, device=x.device)
    regularizer[-1, -1] = 0.0
    return torch.linalg.solve(gram + float(ridge) * regularizer, cross)


def weighted_action_mse(
    predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> float:
    row_mse = (predicted - target).square().mean(dim=1)
    normalized = weights / weights.sum()
    return float((row_mse * normalized).sum())


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        LC068_REPORT: LC068_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC069 bound input changed: {path}")
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC068_REPORT.read_text())
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        parent_report.get("schema") != "vq2_lc062_phase2_bias_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent.get("schema") != "vq2_lc062_phase2_bias_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or dataset.get("schema") != "vq2_lc028_index2_student_dagger_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 431_450
        or dataset.get("teacher_plant_actions_executed") != 0
        or rejected.get("schema") != "vq2_lc068_phase2_failure_full_course_report_v1"
        or rejected.get("selected_candidate") is not None
        or rejected.get("numerically_admitted")
        or not rejected.get("diagnostic_valid")
        or rejected.get("items", [{}])[0].get("gate3_passes") != 34
        or rejected.get("items", [{}, {}])[1].get("gate3_passes") != 33
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC062/LC068 and LC028 do not authorize LC069")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT, LC068_REPORT,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/collect_vq2_vg062_warmed_teacher_intervention_features.py",
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
        raise RuntimeError("LC069 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC069 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    device = torch.device(device_name)
    started = time.perf_counter()

    records = np.memmap(FEATURES, dtype=FEATURE_DTYPE, mode="r")
    agents = np.asarray(records["agent_index"], dtype=np.int64)
    steps = np.asarray(records["step"], dtype=np.int64)
    terminals = np.asarray(records["terminal"], dtype=np.uint8)
    present, outcome = terminal_outcome_by_agent(agents, steps, terminals)
    if present.size != 248 or int((outcome[present] == 1).sum()) != 57:
        raise RuntimeError("LC069 source-locked outcome counts changed")
    train_agents, validation_agents = stratified_agent_split(present, outcome)
    train_weights_np = trajectory_class_weights(agents, outcome, train_agents)
    validation_weights_np = trajectory_class_weights(agents, outcome, validation_agents)
    train_mask_np = train_weights_np > 0
    validation_mask_np = validation_weights_np > 0

    state = parent["model_state"]
    input_weight = state["indexed_phase_residual_input"][TARGET_PHASE].to(device)
    input_bias = state["indexed_phase_residual_input_bias"][TARGET_PHASE].to(device)
    output_weight = state["indexed_phase_residual_output"][TARGET_PHASE].to(device)
    output_bias = state["indexed_phase_residual_output_bias"][TARGET_PHASE].to(device)
    action_weight = state["action_head.weight"].to(device)
    action_bias = state["action_head.bias"].to(device)
    feature_values = np.empty((records.size, input_weight.shape[0]), dtype=np.float32)
    base_values = np.empty((records.size, action_weight.shape[0]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, records.size, FEATURE_CHUNK):
            stop = min(records.size, start + FEATURE_CHUNK)
            hidden = torch.from_numpy(
                np.asarray(records["hidden"][start:stop], dtype=np.float32)
            ).to(device)
            feature_values[start:stop] = torch.tanh(
                hidden @ input_weight.T + input_bias
            ).cpu().numpy()
            base_values[start:stop] = (hidden @ action_weight.T + action_bias).cpu().numpy()

    features = torch.from_numpy(feature_values).to(device)
    base_pre_tanh = torch.from_numpy(base_values).to(device)
    teacher = torch.from_numpy(
        np.array(records["teacher_action"], dtype=np.float32, copy=True)
    ).to(device)
    parent_residual = features @ output_weight.T + output_bias
    failure = torch.from_numpy((outcome[agents] == 0)).to(device)
    target_residual = asymmetric_residual_target(
        base_pre_tanh, parent_residual, teacher, failure
    )
    train_index = torch.from_numpy(np.flatnonzero(train_mask_np)).to(device)
    validation_index = torch.from_numpy(np.flatnonzero(validation_mask_np)).to(device)
    train_features = features.index_select(0, train_index)
    train_targets = target_residual.index_select(0, train_index)
    train_weights = torch.from_numpy(train_weights_np[train_mask_np]).to(device)
    validation_features = features.index_select(0, validation_index)
    validation_base = base_pre_tanh.index_select(0, validation_index)
    validation_teacher = teacher.index_select(0, validation_index)
    validation_parent_residual = parent_residual.index_select(0, validation_index)
    validation_parent_action = torch.tanh(validation_base + validation_parent_residual)
    validation_failure = failure.index_select(0, validation_index)
    validation_weights = torch.from_numpy(validation_weights_np[validation_mask_np]).to(device)

    solutions = [
        weighted_ridge(train_features, train_targets, train_weights, ridge)
        for ridge in RIDGES
    ]
    failure_parent_mse = weighted_action_mse(
        validation_parent_action[validation_failure],
        validation_teacher[validation_failure],
        validation_weights[validation_failure],
    )
    grid: list[dict[str, float]] = []
    for ridge, solution in zip(RIDGES, solutions):
        candidate_residual = validation_features.double() @ solution[:-1] + solution[-1]
        candidate_action = torch.tanh(validation_base.double() + candidate_residual)
        failure_mse = weighted_action_mse(
            candidate_action[validation_failure],
            validation_teacher.double()[validation_failure],
            validation_weights[validation_failure],
        )
        success_drift = weighted_action_mse(
            candidate_action[~validation_failure],
            validation_parent_action.double()[~validation_failure],
            validation_weights[~validation_failure],
        )
        grid.append({
            "ridge": ridge,
            "failure_teacher_action_mse": failure_mse,
            "failure_improvement_factor": failure_parent_mse / max(failure_mse, 1e-20),
            "success_parent_action_drift_mse": success_drift,
            "endpoint_l2": float(solution.norm()),
        })
    eligible = [
        index for index, item in enumerate(grid)
        if math.isfinite(item["failure_teacher_action_mse"])
        and item["success_parent_action_drift_mse"] <= MAXIMUM_SUCCESS_ACTION_DRIFT_MSE
    ]
    selected_index = min(
        eligible or list(range(len(grid))),
        key=lambda index: grid[index]["failure_teacher_action_mse"],
    )
    selected = grid[selected_index]
    solution = solutions[selected_index].detach().cpu().float()
    endpoint_weight = solution[:-1].T.contiguous()
    endpoint_bias = solution[-1].contiguous()
    admitted = bool(
        selected["failure_improvement_factor"] >= MINIMUM_FAILURE_IMPROVEMENT
        and selected["success_parent_action_drift_mse"] <= MAXIMUM_SUCCESS_ACTION_DRIFT_MSE
        and torch.isfinite(endpoint_weight).all()
        and torch.isfinite(endpoint_bias).all()
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "numerically_admitted": admitted,
        "target_phase": TARGET_PHASE,
        "output_weight": endpoint_weight,
        "output_bias": endpoint_bias,
        "parent_output_weight": output_weight.detach().cpu(),
        "parent_output_bias": output_bias.detach().cpu(),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "selected_ridge": selected["ridge"],
        "validation": selected,
    }
    checkpoint_path = output / "decoder_endpoint.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "wall_time_seconds": time.perf_counter() - started,
        "feature_records": int(records.size), "query_agents": int(present.size),
        "success_agents": int((outcome[present] == 1).sum()),
        "failure_agents": int((outcome[present] == 0).sum()),
        "train_agents": int(train_agents.size),
        "validation_agents": int(validation_agents.size),
        "ridge_grid": grid, "selected": selected,
        "failure_parent_teacher_action_mse": failure_parent_mse,
        "minimum_failure_improvement_factor": MINIMUM_FAILURE_IMPROVEMENT,
        "maximum_success_action_drift_mse": MAXIMUM_SUCCESS_ACTION_DRIFT_MSE,
        "source_identity": identity,
        "safety": {
            "training_only_outcome_labels": True,
            "training_only_teacher_targets": True,
            "runtime_privileged_values": 0, "runtime_teacher_actions": 0,
            "student_updates": 0, "flight_sim_packets_sent": 0,
            "submission_authorized": False,
        },
        "next_authority": (
            "Run one small paired teacher-free Gate-3 interpolation screen between LC062 and this endpoint."
            if admitted else "Reject the endpoint and retain LC062."
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
