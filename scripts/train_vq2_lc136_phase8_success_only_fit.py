#!/usr/bin/env python3
"""Fit phase 8 only on LC134's three oracle-rescued trajectories."""

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
    PARAMETER_NAMES, parameter_delta_l2, residual_action,
)
from scripts.train_vq2_lc123_phase6_success_rescue_anchor import interpolate_parameters
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_lc136_phase8_success_only_fit_001"
SCHEMA = "vq2_lc136_phase8_success_only_fit_report_v1"
CHECKPOINT_SCHEMA = "vq2_lc136_phase8_success_only_fit_checkpoint_v1"
SEED = 432_360
TARGET_PHASE = 8
SUCCESS_AGENTS = (143, 148, 175)
CONTROL_AGENTS = (15, 20, 47)
TRAINING_SUCCESS_COUNT = 2
OPTIMIZER_STEPS = 512
BATCH_SIZE = 4_096
EVALUATION_INTERVAL = 16
LEARNING_RATE = 1e-3
ANCHOR_COEFFICIENT = 1e-3
GRADIENT_CLIP = 1.0
TARGET_ACTION_CLIP = 0.999
SCALES = (0.10, 0.30, 0.50, 1.0)
MINIMUM_SUCCESS_IMPROVEMENT = 2.0
MAXIMUM_CONTROL_PARENT_DRIFT_MSE = 0.02
MAXIMUM_PHASE_DELTA_L2 = 64.0
FROZEN_STATE_FIELD = "frozen_non_phase8_state_exact"
FIT_METADATA_FIELD = "success_only_phase8_fit"
PARENT_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc123_phase6_success_rescue_anchor_001"
)
PARENT_CHECKPOINT = PARENT_DIR / "policy_selected.pt"
PARENT_CHECKPOINT_SHA256 = "0d0d9f6ba796dbf1803521214a0c030a79673f24cff538ae05338183ded4f558"
PARENT_REPORT = PARENT_DIR / "report.json"
PARENT_REPORT_SHA256 = "92f29cdd0f3c76b2c031be3acdcf5bc23fecc9c7e499cafe3cf6edc5758c0bf9"
DATASET_DIR = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc134_phase8_paired_anchor_rescue_001"
)
FEATURES = DATASET_DIR / "features.bin"
FEATURES_SHA256 = "e9454912542d6f42626f4763ea500b55ca618d201bb6a8eff85280b13da83f95"
DATASET_REPORT = DATASET_DIR / "report.json"
DATASET_REPORT_SHA256 = "570e11e8b16f0191e64a4dc462349e31f6707c71eb2a17c2139ebd2cfde975a2"
LC135_REPORT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_lc135_phase8_paired_rescue_fit_001/report.json"
)
LC135_REPORT_SHA256 = "61553ee523d99dfabe684a812a8e95ba6163ce9752c83cf87557d053e73843d4"
PREREGISTRATION = ROOT / "docs/vq2_lc136_phase8_success_only_fit_preregistration_2026-08-01.md"
RUNNER = ROOT / "scripts/run_vq2_lc136_vast.sh"
TEST = ROOT / "tests/test_train_vq2_lc136_phase8_success_only_fit.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def trajectory_weights(row_agents: np.ndarray, selected_agents: np.ndarray) -> np.ndarray:
    row_agents = np.asarray(row_agents, dtype=np.int64)
    selected_agents = np.asarray(selected_agents, dtype=np.int64)
    result = np.zeros(row_agents.shape, dtype=np.float64)
    counts = np.bincount(row_agents, minlength=int(row_agents.max(initial=0)) + 1)
    for agent in selected_agents:
        if agent >= counts.size or counts[agent] == 0:
            raise ValueError("selected LC136 trajectory has no rows")
        result[row_agents == agent] = 1.0 / (selected_agents.size * counts[agent])
    if not np.isclose(result.sum(), 1.0, atol=1e-10):
        raise RuntimeError("LC136 trajectory weights do not sum to one")
    return result


def verify_inputs() -> dict[str, Any]:
    for path, digest in {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        LC135_REPORT: LC135_REPORT_SHA256,
    }.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"LC136 bound input changed: {path}")
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    parent_report = json.loads(PARENT_REPORT.read_text())
    dataset = json.loads(DATASET_REPORT.read_text())
    rejected = json.loads(LC135_REPORT.read_text())
    if (
        parent.get("schema")
        != "vq2_lc123_phase6_success_rescue_anchor_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or not parent_report.get("numerically_admitted")
        or parent_report.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or dataset.get("schema") != "vq2_lc134_phase8_paired_anchor_rescue_report_v1"
        or not dataset.get("training_dataset_admitted")
        or dataset.get("query_outcome_success_agents") != [143, 148, 175]
        or dataset.get("query_outcome_failure_agents") != [15, 20, 47]
        or rejected.get("schema") != "vq2_lc135_phase8_paired_rescue_fit_report_v1"
        or rejected.get("numerically_admitted")
        or rejected.get("selected", {}).get("success_improvement_factor", 2.0) >= 1.20
        or rejected.get("safety", {}).get("flight_sim_packets_sent") != 0
        or rejected.get("safety", {}).get("submission_authorized")
    ):
        raise RuntimeError("LC123/LC134/LC135 do not authorize LC136")
    return parent


def source_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        PARENT_CHECKPOINT, PARENT_REPORT, FEATURES, DATASET_REPORT, LC135_REPORT,
        ROOT / "scripts/collect_vq2_lc134_phase8_paired_anchor_rescue.py",
        ROOT / "scripts/train_vq2_lc135_phase8_paired_rescue_fit.py",
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
        raise RuntimeError("LC136 preregisters CUDA fitting")
    report_path = output / "report.json"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        return json.loads(report_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("LC136 output exists without a terminal report")
    output.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    started = time.perf_counter()
    device = torch.device(device_name)

    records = np.memmap(FEATURES, dtype=FEATURE_DTYPE, mode="r")
    phase = np.asarray(records["phase_index"], dtype=np.int64) == TARGET_PHASE
    success_agents = np.asarray(SUCCESS_AGENTS, dtype=np.int64)
    control_agents = np.asarray(CONTROL_AGENTS, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    shuffled = success_agents.copy()
    rng.shuffle(shuffled)
    training_agents = np.sort(shuffled[:TRAINING_SUCCESS_COUNT])
    validation_agents = np.sort(shuffled[TRAINING_SUCCESS_COUNT:])
    if not len(training_agents) or not len(validation_agents):
        raise RuntimeError("success split requires nonempty training and validation sets")
    success_rows = np.flatnonzero(phase & np.isin(records["agent_index"], success_agents))
    control_rows = np.flatnonzero(phase & np.isin(records["agent_index"], control_agents))
    success = records[success_rows]
    control = records[control_rows]
    success_row_agents = np.asarray(success["agent_index"], dtype=np.int64)
    control_row_agents = np.asarray(control["agent_index"], dtype=np.int64)
    train_weights_np = trajectory_weights(success_row_agents, training_agents)
    validation_weights_np = trajectory_weights(success_row_agents, validation_agents)
    control_weights_np = trajectory_weights(control_row_agents, control_agents)
    train_index = np.flatnonzero(train_weights_np > 0)
    validation_index = np.flatnonzero(validation_weights_np > 0)

    state = {name: value.detach().cpu().clone() for name, value in parent["model_state"].items()}
    parent_state_hash = state_sha256(state)
    hidden = torch.from_numpy(np.array(success["hidden"], dtype=np.float32, copy=True)).to(device)
    teacher = torch.from_numpy(
        np.array(success["teacher_action"], dtype=np.float32, copy=True)
    ).to(device).clamp(-TARGET_ACTION_CLIP, TARGET_ACTION_CLIP)
    control_hidden = torch.from_numpy(
        np.array(control["hidden"], dtype=np.float32, copy=True)
    ).to(device)
    action_weight = state["action_head.weight"].to(device)
    action_bias = state["action_head.bias"].to(device)
    base_pre = hidden @ action_weight.T + action_bias
    control_base = control_hidden @ action_weight.T + action_bias
    parents = tuple(
        state[name][TARGET_PHASE].to(device).detach().clone() for name in PARAMETER_NAMES
    )
    with torch.no_grad():
        parent_validation = residual_action(hidden, base_pre, parents)
        control_parent = residual_action(control_hidden, control_base, parents)

    train_index_tensor = torch.from_numpy(train_index).to(device)
    train_probabilities = torch.from_numpy(train_weights_np[train_index].astype(np.float32)).to(device)
    train_probabilities /= train_probabilities.sum()
    validation_index_tensor = torch.from_numpy(validation_index).to(device)
    validation_weights = torch.from_numpy(
        validation_weights_np[validation_index].astype(np.float64)
    ).to(device)
    control_weights = torch.from_numpy(control_weights_np.astype(np.float64)).to(device)
    parent_validation_mse = regression.weighted_action_mse(
        parent_validation.index_select(0, validation_index_tensor),
        teacher.index_select(0, validation_index_tensor), validation_weights,
    )

    parameters = tuple(value.clone().requires_grad_(True) for value in parents)
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_parameters = tuple(value.detach().clone() for value in parents)
    for step in range(1, OPTIMIZER_STEPS + 1):
        sampled = torch.multinomial(
            train_probabilities, BATCH_SIZE, replacement=True, generator=generator
        )
        index = train_index_tensor.index_select(0, sampled)
        prediction = residual_action(
            hidden.index_select(0, index), base_pre.index_select(0, index), parameters
        )
        action_loss = (prediction - teacher.index_select(0, index)).square().mean()
        anchor = sum(
            (value - original).square().mean()
            for value, original in zip(parameters, parents)
        )
        loss = action_loss + ANCHOR_COEFFICIENT * anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
        optimizer.step()
        if step % EVALUATION_INTERVAL and step != OPTIMIZER_STEPS:
            continue
        scale_items: list[dict[str, Any]] = []
        with torch.no_grad():
            for scale in SCALES:
                candidate = interpolate_parameters(parents, parameters, scale)
                validation_action = residual_action(
                    hidden.index_select(0, validation_index_tensor),
                    base_pre.index_select(0, validation_index_tensor), candidate,
                )
                validation_mse = regression.weighted_action_mse(
                    validation_action, teacher.index_select(0, validation_index_tensor),
                    validation_weights,
                )
                control_action = residual_action(control_hidden, control_base, candidate)
                control_drift = regression.weighted_action_mse(
                    control_action, control_parent, control_weights,
                )
                delta = float(parameter_delta_l2(candidate, parents))
                item = {
                    "step": step, "scale": scale,
                    "validation_success_teacher_action_mse": validation_mse,
                    "success_improvement_factor": parent_validation_mse / max(validation_mse, 1e-20),
                    "paired_control_parent_action_drift_mse": control_drift,
                    "parameter_delta_l2": delta,
                }
                scale_items.append(item)
                eligible = bool(
                    math.isfinite(validation_mse) and math.isfinite(control_drift)
                    and control_drift <= MAXIMUM_CONTROL_PARENT_DRIFT_MSE
                    and delta <= MAXIMUM_PHASE_DELTA_L2
                )
                if eligible and (best is None or validation_mse < best["validation_success_teacher_action_mse"]):
                    best = dict(item)
                    best_parameters = tuple(value.detach().clone() for value in candidate)
        history.append({
            "step": step,
            "training_success_teacher_action_mse": float(action_loss.detach()),
            "scales": scale_items,
        })

    if best is None:
        raise RuntimeError("LC136 found no finite bounded candidate")
    admitted = bool(
        best["success_improvement_factor"] >= MINIMUM_SUCCESS_IMPROVEMENT
        and best["paired_control_parent_action_drift_mse"] <= MAXIMUM_CONTROL_PARENT_DRIFT_MSE
        and best["parameter_delta_l2"] <= MAXIMUM_PHASE_DELTA_L2
        and all(bool(torch.isfinite(value).all()) for value in best_parameters)
    )
    for name, value in zip(PARAMETER_NAMES, best_parameters):
        state[name][TARGET_PHASE].copy_(value.detach().cpu().float())
    frozen_exact = all(
        torch.equal(
            state[name][torch.arange(state[name].shape[0]) != TARGET_PHASE],
            parent["model_state"][name][torch.arange(state[name].shape[0]) != TARGET_PHASE],
        ) if name in PARAMETER_NAMES else torch.equal(state[name], parent["model_state"][name])
        for name in state
    )
    admitted &= frozen_exact
    candidate_state_hash = state_sha256(state)
    checkpoint = {
        **parent, "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "model_state": state, "optimizer_updates": OPTIMIZER_STEPS,
        "numerically_admitted": admitted, "deployment_candidate": False,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        FIT_METADATA_FIELD: {"selected": best, "target_phase": TARGET_PHASE},
    }
    checkpoint_path = output / "policy_selected.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted, "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_state_sha256": parent_state_hash,
        "candidate_state_sha256": candidate_state_hash,
        "phase": TARGET_PHASE, "success_records": int(success.size),
        "control_diagnostic_records": int(control.size),
        "training_agents": training_agents.tolist(),
        "validation_agents": validation_agents.tolist(),
        "parent_validation_success_teacher_action_mse": parent_validation_mse,
        "selected": best, "history": history,
        "optimizer_steps": OPTIMIZER_STEPS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "anchor_coefficient": ANCHOR_COEFFICIENT,
        "minimum_success_improvement": MINIMUM_SUCCESS_IMPROVEMENT,
        "maximum_control_parent_drift_mse": MAXIMUM_CONTROL_PARENT_DRIFT_MSE,
        FROZEN_STATE_FIELD: frozen_exact,
        "wall_time_seconds": time.perf_counter() - started,
        "source_identity": identity,
        "safety": {
            "training_only_teacher_targets": True, "runtime_teacher_actions": 0,
            "flight_sim_packets_sent": 0, "submission_authorized": False,
        },
        "next_authority": (
            "Run exactly one reduced LC123-versus-LC136 teacher-free raw-9 screen; no FlightSim authority."
            if admitted else "Reject LC136 and retain LC105; do not run FlightSim."
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
