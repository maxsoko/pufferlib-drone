#!/usr/bin/env python3
"""Fit a constrained success-anchored phase-6 Puffer decoder endpoint."""

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
import scripts.train_vq2_lc069_phase2_failure_teacher_endpoint as regression
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc080_phase6_success_anchored_endpoint_001"
SCHEMA = "vq2_lc080_phase6_success_anchored_endpoint_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc080_phase6_success_anchored_endpoint_checkpoint_v1"
SEED = 431800
TARGET_PHASE = 6
FEATURE_CHUNK = 65_536
TARGET_ACTION_CLIP = 0.999
RIDGES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
ALPHAS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075,
          0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
MINIMUM_FAILURE_IMPROVEMENT = 1.01
MAXIMUM_SUCCESS_ACTION_DRIFT_MSE = 0.00025
EXPECTED_QUERY_AGENTS = 35
EXPECTED_SUCCESS_AGENTS = 2
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
    / "vq2_lc079_phase6_uncensored_outcomes_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "2a1dcf1be9c7e983a3e5e36a62a5bad36571d9ca0175d2ddde014760047fb4be"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "c09030054d28caeaaad370531d93e77595adecde2d2a726c5695b4b7d0a13a31"
PREREGISTRATION = ROOT / "docs/vq2_lc080_phase6_success_anchored_endpoint_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc080_vast.sh"
TEST = ROOT / "tests/test_fit_vq2_lc080_phase6_success_anchored_endpoint.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
EXTRA_SOURCE_PATHS: tuple[Path, ...] = ()


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


def interpolate_endpoint(
    parent_weight: torch.Tensor,
    parent_bias: torch.Tensor,
    endpoint_weight: torch.Tensor,
    endpoint_bias: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        parent_weight + float(alpha) * (endpoint_weight - parent_weight),
        parent_bias + float(alpha) * (endpoint_bias - parent_bias),
    )


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC080 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or parent_report.get("schema") != "vq2_lc073_phase2_constrained_endpoint_full_course_report_v1"
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema") != "vq2_lc079_phase6_uncensored_outcomes_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 67_736
        or dataset.get("query_agents") != 35
        or dataset.get("query_outcome_success_agents") != 2
        or dataset.get("query_outcome_failure_agents") != 33
        or dataset.get("query_outcome_censored_agents") != 0
        or dataset.get("teacher_plant_actions_executed") != 0
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC073/LC079 do not authorize LC080")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_vg062_warmed_teacher_intervention_features.py",
        ROOT / "scripts/train_vq2_lc069_phase2_failure_teacher_endpoint.py",
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        *EXTRA_SOURCE_PATHS,
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
        raise RuntimeError("LC080 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC080 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    device = torch.device(device_name)
    started = time.perf_counter()

    records = np.memmap(FEATURES, dtype=FEATURE_DTYPE, mode="r")
    agents = np.asarray(records["agent_index"], dtype=np.int64)
    steps = np.asarray(records["step"], dtype=np.int64)
    terminals = np.asarray(records["terminal"], dtype=np.uint8)
    present, outcome = regression.terminal_outcome_by_agent(agents, steps, terminals)
    if (
        present.size != EXPECTED_QUERY_AGENTS
        or int((outcome[present] == 1).sum()) != EXPECTED_SUCCESS_AGENTS
    ):
        raise RuntimeError("LC080 source-locked uncensored outcome counts changed")
    train_agents, validation_agents = stratified_agent_split(present, outcome)
    train_weights_np = regression.trajectory_class_weights(agents, outcome, train_agents)
    validation_weights_np = regression.trajectory_class_weights(
        agents, outcome, validation_agents
    )
    train_mask_np = train_weights_np > 0
    validation_mask_np = validation_weights_np > 0

    state = parent["model_state"]
    input_weight = state["indexed_phase_residual_input"][TARGET_PHASE].to(device)
    input_bias = state["indexed_phase_residual_input_bias"][TARGET_PHASE].to(device)
    parent_weight = state["indexed_phase_residual_output"][TARGET_PHASE].to(device)
    parent_bias = state["indexed_phase_residual_output_bias"][TARGET_PHASE].to(device)
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
    parent_residual = features @ parent_weight.T + parent_bias
    failure = torch.from_numpy((outcome[agents] == 0)).to(device)
    teacher_residual = torch.atanh(
        teacher.clamp(-TARGET_ACTION_CLIP, TARGET_ACTION_CLIP)
    ) - base_pre_tanh
    target_residual = torch.where(failure[:, None], teacher_residual, parent_residual)
    train_index = torch.from_numpy(np.flatnonzero(train_mask_np)).to(device)
    validation_index = torch.from_numpy(np.flatnonzero(validation_mask_np)).to(device)
    train_features = features.index_select(0, train_index)
    train_targets = target_residual.index_select(0, train_index)
    train_weights = torch.from_numpy(train_weights_np[train_mask_np]).to(device)
    validation_features = features.index_select(0, validation_index)
    validation_base = base_pre_tanh.index_select(0, validation_index)
    validation_teacher = teacher.index_select(0, validation_index)
    validation_failure = failure.index_select(0, validation_index)
    validation_weights = torch.from_numpy(
        validation_weights_np[validation_mask_np]
    ).to(device)
    validation_parent_action = torch.tanh(
        validation_base
        + validation_features @ parent_weight.T
        + parent_bias
    )
    parent_failure_mse = regression.weighted_action_mse(
        validation_parent_action[validation_failure],
        validation_teacher[validation_failure],
        validation_weights[validation_failure],
    )

    rows: list[tuple[dict[str, float], torch.Tensor, torch.Tensor]] = []
    for ridge in RIDGES:
        solution = regression.weighted_ridge(
            train_features, train_targets, train_weights, ridge
        )
        raw_weight = solution[:-1].T.to(dtype=parent_weight.dtype)
        raw_bias = solution[-1].to(dtype=parent_bias.dtype)
        for alpha in ALPHAS:
            weight, bias = interpolate_endpoint(
                parent_weight, parent_bias, raw_weight, raw_bias, alpha
            )
            action = torch.tanh(validation_base + validation_features @ weight.T + bias)
            failure_mse = regression.weighted_action_mse(
                action[validation_failure],
                validation_teacher[validation_failure],
                validation_weights[validation_failure],
            )
            success_drift = regression.weighted_action_mse(
                action[~validation_failure],
                validation_parent_action[~validation_failure],
                validation_weights[~validation_failure],
            )
            item = {
                "ridge": ridge,
                "alpha": alpha,
                "failure_teacher_action_mse": failure_mse,
                "failure_improvement_factor": parent_failure_mse / max(failure_mse, 1e-20),
                "success_parent_action_drift_mse": success_drift,
                "parameter_delta_l2": float(torch.sqrt(
                    (weight - parent_weight).double().square().sum()
                    + (bias - parent_bias).double().square().sum()
                )),
            }
            rows.append((item, weight, bias))
    eligible = [
        row for row in rows
        if row[0]["alpha"] > 0.0
        and math.isfinite(row[0]["failure_teacher_action_mse"])
        and row[0]["success_parent_action_drift_mse"]
        <= MAXIMUM_SUCCESS_ACTION_DRIFT_MSE
    ]
    selected_item, selected_weight, selected_bias = min(
        eligible or [row for row in rows if row[0]["alpha"] == 0.0],
        key=lambda row: row[0]["failure_teacher_action_mse"],
    )
    admitted = bool(
        selected_item["alpha"] > 0.0
        and selected_item["failure_improvement_factor"] >= MINIMUM_FAILURE_IMPROVEMENT
        and selected_item["success_parent_action_drift_mse"]
        <= MAXIMUM_SUCCESS_ACTION_DRIFT_MSE
        and torch.isfinite(selected_weight).all()
        and torch.isfinite(selected_bias).all()
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "numerically_admitted": admitted,
        "target_phase": TARGET_PHASE,
        "output_weight": selected_weight.detach().cpu().float(),
        "output_bias": selected_bias.detach().cpu().float(),
        "parent_output_weight": parent_weight.detach().cpu(),
        "parent_output_bias": parent_bias.detach().cpu(),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "selected": selected_item,
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
        "train_success_agents": int((outcome[train_agents] == 1).sum()),
        "validation_success_agents": int((outcome[validation_agents] == 1).sum()),
        "failure_parent_teacher_action_mse": parent_failure_mse,
        "grid": [row[0] for row in rows], "selected": selected_item,
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
            "Run one small fresh-seed paired teacher-free raw-index-7 milestone screen."
            if admitted else "Reject the endpoint and retain LC073."
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
