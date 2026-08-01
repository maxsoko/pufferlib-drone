#!/usr/bin/env python3
"""Fit phase 6 only toward oracle-rescued outcomes and anchor failed ones."""

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

from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import FEATURE_DTYPE
from scripts.eval_vq2_lc015_restore_vg071_head1 import state_sha256
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_once
import scripts.train_vq2_lc069_phase2_failure_teacher_endpoint as regression
from scripts.train_vq2_lc111_phase8_9_full_residual_endpoint import (
    PARAMETER_NAMES,
    parameter_delta_l2,
    residual_action,
)
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc123_phase6_success_rescue_anchor_001"
SCHEMA = "vq2_lc123_phase6_success_rescue_anchor_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
SEED = 432_230
TARGET_PHASE = 6
OPTIMIZER_STEPS = 512
BATCH_SIZE = 4_096
EVALUATION_INTERVAL = 16
LEARNING_RATE = 1e-3
ANCHOR_COEFFICIENT = 1e-3
GRADIENT_CLIP = 1.0
TARGET_ACTION_CLIP = 0.999
SCALES = (0.10, 0.30, 0.50, 1.0)
MINIMUM_SUCCESS_IMPROVEMENT = 1.20
MAXIMUM_FAILURE_PARENT_DRIFT_MSE = 0.00025
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
    / "vq2_lc119_phase6_9_exact_milestone_features_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "54615e1fbbdf3bda4078566918e5f3c7ed72f9e863c548ea33af681c0bf67c89"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "73a5d96e11fd6eea7ad5650b1c384b6b820a1aae50318a94d5503e9b9ef64769"
LC122_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc122_phase6_8_composite_scale_screen_001/report.json"
)
LC122_REPORT_SHA256 = "2ccf30be7ccb3773f2bd9c4a5441562f96f5b5af15c664a154b2ff3afe47f2bc"
PREREGISTRATION = ROOT / "docs/vq2_lc123_phase6_success_rescue_anchor_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc123_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc123_phase6_success_rescue_anchor.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def stratified_agent_split(
    present: np.ndarray, outcome: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    training: list[int] = []
    validation: list[int] = []
    for label in (0, 1):
        agents = present[outcome[present] == label].copy()
        if agents.size < 2:
            raise RuntimeError("LC123 requires both outcome classes to be splittable")
        rng.shuffle(agents)
        cut = max(1, min(agents.size - 1, int(round(0.8 * agents.size))))
        training.extend(agents[:cut].tolist())
        validation.extend(agents[cut:].tolist())
    return (
        np.asarray(sorted(training), dtype=np.int64),
        np.asarray(sorted(validation), dtype=np.int64),
    )


def anchored_target(
    teacher: torch.Tensor, parent_action: torch.Tensor, success: torch.Tensor
) -> torch.Tensor:
    if teacher.shape != parent_action.shape or success.shape != teacher.shape[:-1]:
        raise ValueError("LC123 anchored target tensors do not align")
    return torch.where(success[..., None], teacher, parent_action)


def interpolate_parameters(
    parents: tuple[torch.Tensor, ...],
    endpoint: tuple[torch.Tensor, ...],
    scale: float,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        parent + float(scale) * (value - parent)
        for parent, value in zip(parents, endpoint)
    )


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        LC122_REPORT: LC122_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC123 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC122_REPORT.read_text())
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema")
        != "vq2_lc119_phase6_9_exact_milestone_features_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_records") != 60_082
        or dataset.get("query_agents") != 19
        or len(dataset.get("query_outcome_success_agents", [])) != 4
        or len(dataset.get("query_outcome_failure_agents", [])) != 15
        or rejected.get("schema")
        != "vq2_lc122_phase6_8_composite_scale_screen_report_v1"
        or not rejected.get("diagnostic_valid")
        or rejected.get("causal_screen_selected") is not None
        or any(item.get("target_passes") != 0 for item in rejected.get("items", []))
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC119/LC122 do not authorize LC123")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT, LC122_REPORT,
        ROOT / "scripts/collect_vq2_lc119_phase6_9_exact_milestone_features.py",
        ROOT / "scripts/train_vq2_lc120_phase6_9_outcome_balanced_distillation.py",
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
        raise RuntimeError("LC123 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC123 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    started = time.perf_counter()
    device = torch.device(device_name)

    records = np.memmap(FEATURES, dtype=FEATURE_DTYPE, mode="r")
    selected = np.flatnonzero(np.asarray(records["phase_index"]) == TARGET_PHASE)
    phase_records = records[selected]
    row_agents = np.asarray(phase_records["agent_index"], dtype=np.int64)
    all_agents = np.asarray(records["agent_index"], dtype=np.int64)
    all_steps = np.asarray(records["step"], dtype=np.int64)
    all_terminals = np.asarray(records["terminal"], dtype=np.uint8)
    present, outcome = regression.terminal_outcome_by_agent(
        all_agents, all_steps, all_terminals
    )
    if present.size != 19 or int((outcome[present] == 1).sum()) != 4:
        raise RuntimeError("LC123 exact milestone outcomes changed")
    training_agents, validation_agents = stratified_agent_split(present, outcome)
    train_weights_np = regression.trajectory_class_weights(
        row_agents, outcome, training_agents
    )
    validation_weights_np = regression.trajectory_class_weights(
        row_agents, outcome, validation_agents
    )
    train_index = np.flatnonzero(train_weights_np > 0)
    validation_index = np.flatnonzero(validation_weights_np > 0)

    state = {
        name: value.detach().cpu().clone()
        for name, value in parent["model_state"].items()
    }
    parent_state_hash = state_sha256(state)
    hidden = torch.from_numpy(
        np.array(phase_records["hidden"], dtype=np.float32, copy=True)
    ).to(device)
    teacher = torch.from_numpy(
        np.array(phase_records["teacher_action"], dtype=np.float32, copy=True)
    ).to(device).clamp(-TARGET_ACTION_CLIP, TARGET_ACTION_CLIP)
    base = (
        hidden @ state["action_head.weight"].to(device).T
        + state["action_head.bias"].to(device)
    )
    parents = tuple(
        state[name][TARGET_PHASE].to(device).detach().clone()
        for name in PARAMETER_NAMES
    )
    with torch.no_grad():
        parent_action = residual_action(hidden, base, parents)
    success = torch.from_numpy(outcome[row_agents] == 1).to(device)
    target = anchored_target(teacher, parent_action, success)
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
    validation_hidden = hidden.index_select(0, validation_index_tensor)
    validation_base = base.index_select(0, validation_index_tensor)
    validation_teacher = teacher.index_select(0, validation_index_tensor)
    validation_parent = parent_action.index_select(0, validation_index_tensor)
    validation_success = success.index_select(0, validation_index_tensor)
    success_weights = validation_weights[validation_success]
    failure_weights = validation_weights[~validation_success]
    parent_success_teacher_mse = regression.weighted_action_mse(
        validation_parent[validation_success],
        validation_teacher[validation_success],
        success_weights,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_parameters = tuple(value.detach().clone() for value in parents)
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
            (prediction - target.index_select(0, indices)).square()
        )
        anchor = sum(
            (value - original).square().mean()
            for value, original in zip(parameters, parents)
        )
        loss = action_loss + ANCHOR_COEFFICIENT * anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
        optimizer.step()
        if step % EVALUATION_INTERVAL != 0 and step != OPTIMIZER_STEPS:
            continue
        scale_items: list[dict[str, Any]] = []
        with torch.no_grad():
            for scale in SCALES:
                candidate_parameters = interpolate_parameters(
                    parents, parameters, scale
                )
                action = residual_action(
                    validation_hidden, validation_base, candidate_parameters
                )
                success_mse = regression.weighted_action_mse(
                    action[validation_success],
                    validation_teacher[validation_success],
                    success_weights,
                )
                failure_drift = regression.weighted_action_mse(
                    action[~validation_success],
                    validation_parent[~validation_success],
                    failure_weights,
                )
                delta = float(parameter_delta_l2(candidate_parameters, parents))
                item = {
                    "step": step, "scale": scale,
                    "validation_success_teacher_action_mse": success_mse,
                    "success_improvement_factor": (
                        parent_success_teacher_mse / max(success_mse, 1e-20)
                    ),
                    "validation_failure_parent_action_drift_mse": failure_drift,
                    "parameter_delta_l2": delta,
                }
                scale_items.append(item)
                eligible = bool(
                    math.isfinite(success_mse)
                    and math.isfinite(failure_drift)
                    and failure_drift <= MAXIMUM_FAILURE_PARENT_DRIFT_MSE
                    and delta <= MAXIMUM_PHASE_DELTA_L2
                )
                if eligible and (
                    best is None
                    or success_mse < best["validation_success_teacher_action_mse"]
                ):
                    best = dict(item)
                    best_parameters = tuple(
                        value.detach().clone() for value in candidate_parameters
                    )
        history.append({
            "step": step,
            "training_anchored_action_mse": float(action_loss.detach()),
            "scales": scale_items,
        })

    if best is None:
        best = {
            "step": 0, "scale": 0.0,
            "validation_success_teacher_action_mse": parent_success_teacher_mse,
            "success_improvement_factor": 1.0,
            "validation_failure_parent_action_drift_mse": 0.0,
            "parameter_delta_l2": 0.0,
        }
    admitted = bool(
        best["step"] > 0
        and best["scale"] > 0.0
        and best["success_improvement_factor"] >= MINIMUM_SUCCESS_IMPROVEMENT
        and best["validation_failure_parent_action_drift_mse"]
        <= MAXIMUM_FAILURE_PARENT_DRIFT_MSE
        and best["parameter_delta_l2"] <= MAXIMUM_PHASE_DELTA_L2
        and all(bool(torch.isfinite(value).all()) for value in best_parameters)
    )
    for name, value in zip(PARAMETER_NAMES, best_parameters):
        state[name][TARGET_PHASE].copy_(value.detach().cpu().float())
    candidate_state_hash = state_sha256(state)
    checkpoint = {
        **parent, "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "model_state": state, "best_epoch": 0,
        "optimizer_updates": OPTIMIZER_STEPS,
        "numerically_admitted": admitted,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "success_rescue_anchor": {
            "target_phase": TARGET_PHASE,
            "parent_state_sha256": parent_state_hash,
            "candidate_state_sha256": candidate_state_hash,
            "selected": best,
        },
    }
    checkpoint_path = output / "policy_selected.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_state_sha256": parent_state_hash,
        "candidate_state_sha256": candidate_state_hash,
        "phase": TARGET_PHASE,
        "records": int(phase_records.size),
        "agents": int(present.size),
        "success_agents": int((outcome[present] == 1).sum()),
        "failure_agents": int((outcome[present] == 0).sum()),
        "training_agents": training_agents.tolist(),
        "validation_agents": validation_agents.tolist(),
        "parent_validation_success_teacher_action_mse": parent_success_teacher_mse,
        "selected": best, "history": history,
        "optimizer_steps": OPTIMIZER_STEPS,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "anchor_coefficient": ANCHOR_COEFFICIENT,
        "minimum_success_improvement": MINIMUM_SUCCESS_IMPROVEMENT,
        "maximum_failure_parent_drift_mse": MAXIMUM_FAILURE_PARENT_DRIFT_MSE,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "training_only_outcome_labels": True,
            "training_only_teacher_targets": True,
            "runtime_privileged_values": 0, "runtime_teacher_actions": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Run exactly one reduced parent-versus-candidate teacher-free raw-index-10 screen; no FlightSim authority."
            if admitted else "Reject LC123 and retain LC105."
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
