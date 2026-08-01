#!/usr/bin/env python3
"""Outcome-balanced full-residual distillation of LC119 phases 6--9."""

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


TAG = "vq2_lc120_phase6_9_outcome_balanced_distillation_001"
SCHEMA = "vq2_lc120_phase6_9_outcome_balanced_distillation_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc120_phase6_9_outcome_balanced_distillation_checkpoint_v1"
SEED = 432_200
PHASES = (6, 7, 8, 9)
OPTIMIZER_STEPS = 512
BATCH_SIZE = 4_096
EVALUATION_INTERVAL = 16
LEARNING_RATE = 1e-3
ANCHOR_COEFFICIENT = 1e-3
GRADIENT_CLIP = 1.0
TARGET_ACTION_CLIP = 0.999
MINIMUM_OVERALL_IMPROVEMENT = 1.20
MINIMUM_CLASS_IMPROVEMENT = 1.10
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
PREREGISTRATION = ROOT / "docs/vq2_lc120_phase6_9_outcome_balanced_distillation_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc120_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc120_phase6_9_outcome_balanced_distillation.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def stratified_agent_split(
    present: np.ndarray, outcome: np.ndarray, *, phase: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + phase)
    train: list[int] = []
    validation: list[int] = []
    for label in (0, 1):
        ids = present[outcome[present] == label].copy()
        if ids.size < 2:
            raise RuntimeError(f"LC120 phase {phase} lacks a splittable outcome class")
        rng.shuffle(ids)
        cut = max(1, min(ids.size - 1, int(round(0.8 * ids.size))))
        train.extend(ids[:cut].tolist())
        validation.extend(ids[cut:].tolist())
    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(validation), dtype=np.int64),
    )


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC120 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    phases = dataset.get("feature_phase_records", [])
    if (
        parent.get("schema") != "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema")
        != "vq2_lc119_phase6_9_exact_milestone_features_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("feature_sha256") != FEATURES_SHA256
        or dataset.get("feature_records") != 60_082
        or dataset.get("query_agents") != 19
        or len(dataset.get("query_outcome_success_agents", [])) != 4
        or len(dataset.get("query_outcome_failure_agents", [])) != 15
        or len(phases) != 33
        or sum(phases[6:10]) != 60_082
        or sum(phases[:6]) != 0 or sum(phases[10:]) != 0
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
        or dataset.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC105/LC119 do not authorize LC120")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT,
        ROOT / "scripts/collect_vq2_lc119_phase6_9_exact_milestone_features.py",
        ROOT / "scripts/train_vq2_lc111_phase8_9_full_residual_endpoint.py",
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
    outcome: np.ndarray,
    state: dict[str, torch.Tensor],
    phase: int,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    row_agents = np.asarray(phase_records["agent_index"], dtype=np.int64)
    present = np.unique(row_agents)
    train_agents, validation_agents = stratified_agent_split(
        present, outcome, phase=phase
    )
    train_weights_np = regression.trajectory_class_weights(
        row_agents, outcome, train_agents
    )
    validation_weights_np = regression.trajectory_class_weights(
        row_agents, outcome, validation_agents
    )
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
    base_pre_tanh = hidden @ action_weight.T + action_bias
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
    validation_success = torch.from_numpy(
        outcome[row_agents[validation_index]] == 1
    ).to(device)

    validation_hidden = hidden.index_select(0, validation_index_tensor)
    validation_base = base_pre_tanh.index_select(0, validation_index_tensor)
    validation_teacher = teacher.index_select(0, validation_index_tensor)
    with torch.no_grad():
        parent_action = residual_action(
            validation_hidden, validation_base, parents
        )
        parent_overall = regression.weighted_action_mse(
            parent_action, validation_teacher, validation_weights
        )
        parent_success = regression.weighted_action_mse(
            parent_action[validation_success], validation_teacher[validation_success],
            validation_weights[validation_success],
        )
        parent_failure = regression.weighted_action_mse(
            parent_action[~validation_success], validation_teacher[~validation_success],
            validation_weights[~validation_success],
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + phase)
    best_step = 0
    best_score = parent_overall
    best_metrics = (parent_overall, parent_success, parent_failure)
    best_parameters = tuple(value.detach().clone() for value in parents)
    history: list[dict[str, float | int]] = []
    for step in range(1, OPTIMIZER_STEPS + 1):
        sampled = torch.multinomial(
            train_probabilities, BATCH_SIZE, replacement=True, generator=generator
        )
        indices = train_index_tensor.index_select(0, sampled)
        prediction = residual_action(
            hidden.index_select(0, indices),
            base_pre_tanh.index_select(0, indices),
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
                action = residual_action(validation_hidden, validation_base, parameters)
                overall = regression.weighted_action_mse(
                    action, validation_teacher, validation_weights
                )
                success_mse = regression.weighted_action_mse(
                    action[validation_success], validation_teacher[validation_success],
                    validation_weights[validation_success],
                )
                failure_mse = regression.weighted_action_mse(
                    action[~validation_success], validation_teacher[~validation_success],
                    validation_weights[~validation_success],
                )
            history.append({
                "step": step, "training_action_mse": float(action_loss.detach()),
                "validation_teacher_action_mse": overall,
                "validation_success_teacher_action_mse": success_mse,
                "validation_failure_teacher_action_mse": failure_mse,
            })
            if all(map(math.isfinite, (overall, success_mse, failure_mse))) and overall < best_score:
                best_step = step
                best_score = overall
                best_metrics = (overall, success_mse, failure_mse)
                best_parameters = tuple(value.detach().clone() for value in parameters)

    overall, success_mse, failure_mse = best_metrics
    improvement = parent_overall / max(overall, 1e-20)
    success_improvement = parent_success / max(success_mse, 1e-20)
    failure_improvement = parent_failure / max(failure_mse, 1e-20)
    delta_l2 = float(parameter_delta_l2(best_parameters, parents))
    admitted = bool(
        best_step > 0
        and improvement >= MINIMUM_OVERALL_IMPROVEMENT
        and success_improvement >= MINIMUM_CLASS_IMPROVEMENT
        and failure_improvement >= MINIMUM_CLASS_IMPROVEMENT
        and delta_l2 <= MAXIMUM_PHASE_DELTA_L2
        and all(bool(torch.isfinite(value).all()) for value in best_parameters)
    )
    report = {
        "records": int(phase_records.size), "agents": int(present.size),
        "success_agents": int((outcome[present] == 1).sum()),
        "failure_agents": int((outcome[present] == 0).sum()),
        "training_agents": train_agents.tolist(),
        "validation_agents": validation_agents.tolist(),
        "parent_teacher_action_mse": parent_overall,
        "best_validation_teacher_action_mse": overall,
        "overall_improvement_factor": improvement,
        "success_improvement_factor": success_improvement,
        "failure_improvement_factor": failure_improvement,
        "parameter_delta_l2": delta_l2, "best_step": best_step,
        "history": history, "numerically_admitted": admitted,
    }
    return report, tuple(value.detach().cpu().float() for value in best_parameters)


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False
) -> dict[str, Any]:
    parent = verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LC120 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC120 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    started = time.perf_counter()
    device = torch.device(device_name)
    records = np.memmap(FEATURES, dtype=FEATURE_DTYPE, mode="r")
    agents = np.asarray(records["agent_index"], dtype=np.int64)
    steps = np.asarray(records["step"], dtype=np.int64)
    terminals = np.asarray(records["terminal"], dtype=np.uint8)
    present, outcome = regression.terminal_outcome_by_agent(agents, steps, terminals)
    if present.size != 19 or int((outcome[present] == 1).sum()) != 4:
        raise RuntimeError("LC120 exact milestone outcomes changed")
    state = {
        name: value.detach().cpu().clone()
        for name, value in parent["model_state"].items()
    }
    parent_state_hash = state_sha256(state)
    phase_reports: dict[str, Any] = {}
    for phase in PHASES:
        indices = np.flatnonzero(np.asarray(records["phase_index"]) == phase)
        phase_report, parameters = fit_phase(
            records[indices], outcome, state, phase, device=device
        )
        phase_reports[str(phase)] = phase_report
        for name, value in zip(PARAMETER_NAMES, parameters):
            state[name][phase].copy_(value)
    all_admitted = all(
        phase_report["numerically_admitted"]
        for phase_report in phase_reports.values()
    )
    candidate_state_hash = state_sha256(state)
    checkpoint = {
        **parent, "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "model_state": state, "best_epoch": 0,
        "optimizer_updates": OPTIMIZER_STEPS * len(PHASES),
        "numerically_admitted": all_admitted,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "outcome_balanced_full_residual": {
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
        "phase_reports": phase_reports,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "training_only_outcome_labels": True,
            "training_only_teacher_targets": True,
            "runtime_privileged_values": 0, "runtime_teacher_actions": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Run one teacher-free pairwise-256 complete-Puffer scale bracket at raw progress 10."
            if all_admitted else
            "Reject LC120 and retain LC105."
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
