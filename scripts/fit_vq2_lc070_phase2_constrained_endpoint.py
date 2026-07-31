#!/usr/bin/env python3
"""Constrain LC069 along its parent-to-endpoint decoder line."""

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
import scripts.train_vq2_lc069_phase2_failure_teacher_endpoint as lc069
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc070_phase2_constrained_endpoint_001"
SCHEMA = "vq2_lc070_phase2_constrained_endpoint_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc070_phase2_constrained_endpoint_checkpoint_v1"
ALPHAS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30)
MINIMUM_FAILURE_IMPROVEMENT = 1.03
MAXIMUM_SUCCESS_ACTION_DRIFT_MSE = 0.0025
LC069_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc069_phase2_failure_teacher_endpoint_001"
)
LC069_CHECKPOINT = LC069_DIR / "decoder_endpoint.pt"
LC069_CHECKPOINT_SHA256 = "4b0bbbfb80e9af682c5bf6160d95bdd47eaae10dca345fdebd206b9d3da6a931"
LC069_REPORT = LC069_DIR / "report.json"
LC069_REPORT_SHA256 = "c4cf1aa93d3f00a19d7172606e2e2d676a4d61b8e85d4e0c9e6ed8621c6c7dec"
PREREGISTRATION = ROOT / "docs/vq2_lc070_phase2_constrained_endpoint_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_lc070_vast.sh"
TEST = ROOT / "tests/test_fit_vq2_lc070_phase2_constrained_endpoint.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


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


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    parent = lc069.verify_inputs()
    for path, digest in {
        LC069_CHECKPOINT: LC069_CHECKPOINT_SHA256,
        LC069_REPORT: LC069_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC070 bound input changed: {path}")
    endpoint = torch.load(LC069_CHECKPOINT, map_location="cpu", weights_only=False)
    report = json.loads(LC069_REPORT.read_text())
    if (
        endpoint.get("schema") != lc069.CHECKPOINT_SCHEMA
        or endpoint.get("numerically_admitted")
        or endpoint.get("parent_checkpoint_sha256") != lc069.PARENT_CHECKPOINT_SHA256
        or report.get("schema") != lc069.SCHEMA
        or report.get("numerically_admitted")
        or report.get("selected", {}).get("failure_improvement_factor", 0.0) < 2.72
        or report.get("selected", {}).get("success_parent_action_drift_mse", 0.0) < 0.13
        or report.get("safety", {}).get("flight_sim_packets_sent") != 0
        or report.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("rejected LC069 does not authorize constrained analysis")
    return parent, endpoint


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        LC069_CHECKPOINT, LC069_REPORT, lc069.PARENT_CHECKPOINT,
        lc069.PARENT_REPORT, lc069.FEATURES, lc069.DATASET_REPORT,
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
    parent, endpoint = verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC070 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC070 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    device = torch.device(device_name)
    started = time.perf_counter()

    records = np.memmap(lc069.FEATURES, dtype=FEATURE_DTYPE, mode="r")
    agents = np.asarray(records["agent_index"], dtype=np.int64)
    steps = np.asarray(records["step"], dtype=np.int64)
    terminals = np.asarray(records["terminal"], dtype=np.uint8)
    present, outcome = lc069.terminal_outcome_by_agent(agents, steps, terminals)
    _, validation_agents = lc069.stratified_agent_split(present, outcome)
    validation_weights_np = lc069.trajectory_class_weights(
        agents, outcome, validation_agents
    )
    validation_mask_np = validation_weights_np > 0
    validation_indices_np = np.flatnonzero(validation_mask_np)

    state = parent["model_state"]
    input_weight = state["indexed_phase_residual_input"][lc069.TARGET_PHASE].to(device)
    input_bias = state["indexed_phase_residual_input_bias"][lc069.TARGET_PHASE].to(device)
    parent_weight = state["indexed_phase_residual_output"][lc069.TARGET_PHASE].to(device)
    parent_bias = state["indexed_phase_residual_output_bias"][lc069.TARGET_PHASE].to(device)
    endpoint_weight = endpoint["output_weight"].to(device)
    endpoint_bias = endpoint["output_bias"].to(device)
    action_weight = state["action_head.weight"].to(device)
    action_bias = state["action_head.bias"].to(device)
    hidden = torch.from_numpy(
        np.array(records["hidden"][validation_indices_np], dtype=np.float32, copy=True)
    ).to(device)
    features = torch.tanh(hidden @ input_weight.T + input_bias)
    base_pre_tanh = hidden @ action_weight.T + action_bias
    teacher = torch.from_numpy(
        np.array(records["teacher_action"][validation_indices_np], dtype=np.float32, copy=True)
    ).to(device)
    failure = torch.from_numpy((outcome[agents[validation_indices_np]] == 0)).to(device)
    weights = torch.from_numpy(validation_weights_np[validation_mask_np]).to(device)
    parent_residual = features @ parent_weight.T + parent_bias
    parent_action = torch.tanh(base_pre_tanh + parent_residual)
    parent_failure_mse = lc069.weighted_action_mse(
        parent_action[failure], teacher[failure], weights[failure]
    )

    grid: list[dict[str, float]] = []
    endpoint_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    for alpha in ALPHAS:
        weight, bias = interpolate_endpoint(
            parent_weight, parent_bias, endpoint_weight, endpoint_bias, alpha
        )
        endpoint_rows.append((weight, bias))
        action = torch.tanh(base_pre_tanh + features @ weight.T + bias)
        failure_mse = lc069.weighted_action_mse(
            action[failure], teacher[failure], weights[failure]
        )
        success_drift = lc069.weighted_action_mse(
            action[~failure], parent_action[~failure], weights[~failure]
        )
        grid.append({
            "alpha": alpha,
            "failure_teacher_action_mse": failure_mse,
            "failure_improvement_factor": parent_failure_mse / max(failure_mse, 1e-20),
            "success_parent_action_drift_mse": success_drift,
            "parameter_delta_l2": float(torch.sqrt(
                (weight - parent_weight).double().square().sum()
                + (bias - parent_bias).double().square().sum()
            )),
        })
    eligible = [
        index for index, item in enumerate(grid)
        if index > 0
        and math.isfinite(item["failure_teacher_action_mse"])
        and item["success_parent_action_drift_mse"] <= MAXIMUM_SUCCESS_ACTION_DRIFT_MSE
    ]
    selected_index = min(
        eligible or [0], key=lambda index: grid[index]["failure_teacher_action_mse"]
    )
    selected = grid[selected_index]
    selected_weight, selected_bias = endpoint_rows[selected_index]
    admitted = bool(
        selected_index > 0
        and selected["failure_improvement_factor"] >= MINIMUM_FAILURE_IMPROVEMENT
        and selected["success_parent_action_drift_mse"] <= MAXIMUM_SUCCESS_ACTION_DRIFT_MSE
        and torch.isfinite(selected_weight).all()
        and torch.isfinite(selected_bias).all()
    )
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "numerically_admitted": admitted,
        "target_phase": lc069.TARGET_PHASE,
        "output_weight": selected_weight.detach().cpu(),
        "output_bias": selected_bias.detach().cpu(),
        "parent_output_weight": parent_weight.detach().cpu(),
        "parent_output_bias": parent_bias.detach().cpu(),
        "parent_checkpoint_sha256": lc069.PARENT_CHECKPOINT_SHA256,
        "source_endpoint_sha256": LC069_CHECKPOINT_SHA256,
        "selected_alpha": selected["alpha"],
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
        "feature_records": int(records.size),
        "validation_agents": int(validation_agents.size),
        "failure_parent_teacher_action_mse": parent_failure_mse,
        "grid": grid, "selected": selected,
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
            "Run one fresh-seed paired teacher-free Gate-3 interpolation screen from LC062 toward this constrained endpoint."
            if admitted else "Reject the constrained endpoint and retain LC062."
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
