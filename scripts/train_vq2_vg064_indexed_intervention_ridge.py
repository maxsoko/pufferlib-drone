#!/usr/bin/env python3
"""Distill VG063 intervention features into learned public-indexed Puffer heads."""

from __future__ import annotations

import argparse
import json
import math
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

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseResidualActor
from scripts.collect_vq2_vg062_warmed_teacher_intervention_features import (
    FEATURE_DTYPE,
    HIDDEN_SIZE,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_atomic


TAG = "vq2_vg064_indexed_intervention_ridge_001"
SCHEMA = "vq2_vg064_indexed_intervention_ridge_report_v1"
STATE_SCHEMA = "vq2_vg064_indexed_intervention_ridge_state_v1"
CHECKPOINT_SCHEMA = "vq2_vg064_indexed_intervention_ridge_checkpoint_v1"
SEED = 429199
TARGET_PHASES = tuple(range(1, 12))
VALIDATION_AGENT_MODULUS = 8
VALIDATION_AGENT_REMAINDER = 0
CHUNK_ROWS = 65536
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TARGET_ACTION_CLIP = 0.999
MINIMUM_IMPROVEMENT_FACTOR = 1.25
MAXIMUM_HEAD_L2 = 64.0
PARENT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg033_variable_gate_eight_source_refit_001"
)
PARENT_CHECKPOINT = PARENT / "policy_best.pt"
PARENT_CHECKPOINT_SHA256 = "56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26"
PARENT_REPORT = PARENT / "report.json"
PARENT_REPORT_SHA256 = "033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60"
DATASET = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg063_horizon_corrected_intervention_features_001"
)
DATASET_REPORT = DATASET / "report.json"
DATASET_REPORT_SHA256 = "d74f16bdd6d9da17a4fffedf52e53cb34173940d0e6943b755c3c683bd1a583d"
FEATURES = DATASET / "features.bin"
FEATURES_SHA256 = "ccfc07e578e1f16900b9d020c5c9063676058c518b8524bf5452738566181580"
DATASET_ADMISSION = ROOT / "docs/vq2_vg063_horizon_corrected_intervention_admission_2026-07-31.json"
DATASET_ADMISSION_SHA256 = "47c1e107dbf283d49f4b44e968d139a8743a9d3aa194777a23e175099f02e3cc"
GOAL = ROOT / "docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"
GOAL_SHA256 = "03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1"
PREREGISTRATION = ROOT / "docs/vq2_vg064_indexed_intervention_ridge_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg064_vast.sh"
TEST = ROOT / "tests/test_train_vq2_vg064_indexed_intervention_ridge.py"
INFRASTRUCTURE_REPAIR = ROOT / "docs/vq2_vg064_memmap_stride_infrastructure_repair_2026-07-31.json"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        INFRASTRUCTURE_REPAIR, GOAL,
        PARENT_CHECKPOINT, PARENT_REPORT, DATASET_REPORT, FEATURES,
        DATASET_ADMISSION, ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        ROOT / "scripts/collect_vq2_vg062_warmed_teacher_intervention_features.py",
    )


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        DATASET_REPORT: DATASET_REPORT_SHA256,
        FEATURES: FEATURES_SHA256,
        DATASET_ADMISSION: DATASET_ADMISSION_SHA256,
        GOAL: GOAL_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG064 bound input changed: {path}")
    report = json.loads(DATASET_REPORT.read_text())
    admission = json.loads(DATASET_ADMISSION.read_text())
    if (
        report.get("schema") != "vq2_vg063_horizon_corrected_intervention_report_v1"
        or not report.get("training_dataset_admitted")
        or report.get("feature_records") != FEATURES.stat().st_size // FEATURE_DTYPE.itemsize
        or report.get("feature_sha256") != FEATURES_SHA256
        or report.get("metrics", {}).get("env/success_rate") != 1.0
        or report.get("metrics", {}).get("env/crash") != 0.0
    ):
        raise RuntimeError("VG064 dataset report changed")
    if (
        admission.get("schema")
        != "vq2_vg063_horizon_corrected_intervention_admission_v1"
        or not admission.get("training_dataset_admitted")
        or admission.get("dataset", {}).get("feature_sha256") != FEATURES_SHA256
        or admission.get("live_authority")
    ):
        raise RuntimeError("VG064 dataset admission changed")
    return report


def runtime_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "numpy": np.__version__,
        "torch": torch.__version__, "torch_cuda": str(torch.version.cuda),
    }


def source_identity() -> tuple[str, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return commit, {
        str(path.resolve().relative_to(ROOT)): sha256_path(path)
        for path in source_paths()
    }


def validation_mask(agent_index: np.ndarray) -> np.ndarray:
    return (
        np.asarray(agent_index, dtype=np.int64) % VALIDATION_AGENT_MODULUS
        == VALIDATION_AGENT_REMAINDER
    )


def target_pre_tanh_residual(
    base_pre_tanh: torch.Tensor, teacher_action: torch.Tensor,
) -> torch.Tensor:
    clipped = teacher_action.clamp(-TARGET_ACTION_CLIP, TARGET_ACTION_CLIP)
    return torch.atanh(clipped) - base_pre_tanh


def ridge_solution(
    gram: torch.Tensor, cross: torch.Tensor, ridge: float,
) -> torch.Tensor:
    if gram.shape != (HIDDEN_SIZE, HIDDEN_SIZE) or cross.shape != (HIDDEN_SIZE, ACTION_SIZE):
        raise ValueError("VG064 normal-equation shape changed")
    identity = torch.eye(HIDDEN_SIZE, dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + float(ridge) * identity, cross)


def load_parent(device: torch.device) -> tuple[VQ2IndexedPhaseResidualActor, dict[str, Any]]:
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or contract.get("class") != "VQ2PhaseRecurrentActor"
        or contract.get("hidden_size") != HIDDEN_SIZE
        or contract.get("legal_observation_size") != PHASE_LEGAL_OBS_SIZE
    ):
        raise RuntimeError("VG064 parent checkpoint changed")
    model = VQ2IndexedPhaseResidualActor(
        hidden_size=HIDDEN_SIZE, initial_std=float(contract["initial_std"])
    ).to(device)
    model.load_base_state(payload["model_state"])
    model.eval()
    return model, payload


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda", resume: bool = False,
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG064 preregisters CUDA fitting")
    report_path = output / "report.json"
    state_path = output.with_name(f"{output.name}_state.json")
    if report_path.is_file():
        if not resume:
            raise RuntimeError("VG064 terminal output exists; use --resume")
        return json.loads(report_path.read_text())
    dataset_report = verify_inputs()
    commit, hashes = source_identity()
    device = torch.device(device_name)
    model, parent = load_parent(device)
    records = np.memmap(FEATURES, mode="r", dtype=FEATURE_DTYPE)
    count = int(records.shape[0])
    state = {
        "schema": STATE_SCHEMA, "tag": TAG, "status": "running",
        "source_commit": commit, "source_sha256": hashes,
        "runtime": runtime_manifest(), "feature_records": count,
        "seed": SEED, "ridges": list(RIDGES),
        "safety": {"teacher_plant_actions": 0, "flight_sim_packets_sent": 0,
                   "submission_authorized": False},
    }
    write_json_atomic(state_path, state)
    started = time.perf_counter()
    grams = {phase: torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float64) for phase in TARGET_PHASES}
    crosses = {phase: torch.zeros(HIDDEN_SIZE, ACTION_SIZE, dtype=torch.float64) for phase in TARGET_PHASES}
    train_counts = {phase: 0 for phase in TARGET_PHASES}
    validation_counts = {phase: 0 for phase in TARGET_PHASES}

    for start in range(0, count, CHUNK_ROWS):
        chunk = records[start : min(start + CHUNK_ROWS, count)]
        phase_np = np.asarray(chunk["phase_index"], dtype=np.int64)
        validation_np = validation_mask(chunk["agent_index"])
        hidden = torch.from_numpy(
            np.array(chunk["hidden"], dtype=np.float32, copy=True)
        ).to(device)
        base = torch.from_numpy(
            np.array(chunk["base_pre_tanh"], dtype=np.float32, copy=True)
        ).to(device)
        teacher = torch.from_numpy(
            np.array(chunk["teacher_action"], dtype=np.float32, copy=True)
        ).to(device)
        target = target_pre_tanh_residual(base, teacher)
        for phase in TARGET_PHASES:
            phase_mask_np = phase_np == phase
            validation_counts[phase] += int((phase_mask_np & validation_np).sum())
            train_np = phase_mask_np & ~validation_np
            rows = int(train_np.sum())
            train_counts[phase] += rows
            if not rows:
                continue
            index = torch.from_numpy(np.flatnonzero(train_np)).to(device)
            x = hidden.index_select(0, index)
            y = target.index_select(0, index)
            grams[phase] += (x.T @ x).double().cpu()
            crosses[phase] += (x.T @ y).double().cpu()

    solutions: dict[int, list[torch.Tensor]] = {}
    for phase in TARGET_PHASES:
        if train_counts[phase] <= HIDDEN_SIZE or validation_counts[phase] == 0:
            raise RuntimeError(f"VG064 phase {phase} lacks train/validation mass")
        solutions[phase] = [
            ridge_solution(grams[phase], crosses[phase], ridge).float()
            for ridge in RIDGES
        ]

    baseline_sse = {phase: 0.0 for phase in TARGET_PHASES}
    candidate_sse = {phase: np.zeros(len(RIDGES), dtype=np.float64) for phase in TARGET_PHASES}
    for start in range(0, count, CHUNK_ROWS):
        chunk = records[start : min(start + CHUNK_ROWS, count)]
        phase_np = np.asarray(chunk["phase_index"], dtype=np.int64)
        validation_np = validation_mask(chunk["agent_index"])
        hidden = torch.from_numpy(
            np.array(chunk["hidden"], dtype=np.float32, copy=True)
        ).to(device)
        base = torch.from_numpy(
            np.array(chunk["base_pre_tanh"], dtype=np.float32, copy=True)
        ).to(device)
        teacher = torch.from_numpy(
            np.array(chunk["teacher_action"], dtype=np.float32, copy=True)
        ).to(device)
        for phase in TARGET_PHASES:
            chosen_np = (phase_np == phase) & validation_np
            if not chosen_np.any():
                continue
            index = torch.from_numpy(np.flatnonzero(chosen_np)).to(device)
            x = hidden.index_select(0, index)
            b = base.index_select(0, index)
            y = teacher.index_select(0, index)
            baseline_sse[phase] += float(torch.square(torch.tanh(b) - y).sum().item())
            weights = torch.stack([item.to(device) for item in solutions[phase]])
            predicted = torch.tanh(b[None, :, :] + torch.einsum("nh,rho->rno", x, weights))
            errors = torch.square(predicted - y[None, :, :]).sum(dim=(1, 2))
            candidate_sse[phase] += errors.double().cpu().numpy()

    selected_ridge: dict[int, float] = {}
    selected_solution: dict[int, torch.Tensor] = {}
    phase_metrics: dict[str, Any] = {}
    for phase in TARGET_PHASES:
        denominator = validation_counts[phase] * ACTION_SIZE
        baseline_mse = baseline_sse[phase] / denominator
        mse_grid = candidate_sse[phase] / denominator
        best = int(np.argmin(mse_grid))
        selected_ridge[phase] = RIDGES[best]
        selected_solution[phase] = solutions[phase][best]
        phase_metrics[str(phase)] = {
            "train_records": train_counts[phase],
            "validation_records": validation_counts[phase],
            "baseline_action_mse": baseline_mse,
            "ridge_action_mse": {str(ridge): float(mse) for ridge, mse in zip(RIDGES, mse_grid)},
            "selected_ridge": RIDGES[best],
            "selected_action_mse": float(mse_grid[best]),
            "improvement_factor": baseline_mse / max(float(mse_grid[best]), 1e-20),
            "head_l2": float(solutions[phase][best].norm().item()),
        }

    with torch.no_grad():
        for phase, solution in selected_solution.items():
            model.indexed_phase_action_residual[phase].copy_(solution.T.to(device))
    residual = model.indexed_phase_action_residual.detach().cpu()
    parent_exact = all(
        torch.equal(model.state_dict()[name].detach().cpu(), value.detach().cpu())
        for name, value in parent["model_state"].items()
    )
    non_target_zero = bool(
        torch.count_nonzero(residual[0]) == 0
        and torch.count_nonzero(residual[12:]) == 0
    )
    all_improve = all(
        item["improvement_factor"] >= MINIMUM_IMPROVEMENT_FACTOR
        and item["head_l2"] <= MAXIMUM_HEAD_L2
        and math.isfinite(item["selected_action_mse"])
        for item in phase_metrics.values()
    )
    numerically_admitted = bool(parent_exact and non_target_zero and all_improve)
    output.mkdir(parents=True)
    checkpoint = {
        **parent,
        "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "model": {**parent["model"], "class": "VQ2IndexedPhaseResidualActor"},
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "numerically_admitted": numerically_admitted,
        "source_commit": commit, "source_sha256": hashes,
        "dataset_report_sha256": DATASET_REPORT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "dataset_admission_sha256": DATASET_ADMISSION_SHA256,
        "target_phases": list(TARGET_PHASES),
        "selected_ridge": {str(k): v for k, v in selected_ridge.items()},
        "phase_metrics": phase_metrics,
        "safety": {"actor_input_privileged_values": 0, "runtime_teacher_actions": 0,
                   "flight_sim_packets_sent": 0, "submission_authorized": False},
    }
    torch.save(checkpoint, output / "policy_best.pt")
    checkpoint_sha = sha256_path(output / "policy_best.pt")
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": numerically_admitted,
        "source_commit": commit, "source_sha256": hashes,
        "runtime": runtime_manifest(), "wall_time_seconds": time.perf_counter() - started,
        "feature_records": count, "feature_sha256": FEATURES_SHA256,
        "dataset_success_rate": dataset_report["metrics"]["env/success_rate"],
        "validation_split": {"agent_modulus": VALIDATION_AGENT_MODULUS,
                             "validation_remainder": VALIDATION_AGENT_REMAINDER},
        "target_action_clip": TARGET_ACTION_CLIP,
        "ridge_grid": list(RIDGES), "target_phases": list(TARGET_PHASES),
        "phase_metrics": phase_metrics,
        "base_parameters_exact": parent_exact,
        "non_target_heads_zero": non_target_zero,
        "checkpoint": "policy_best.pt", "checkpoint_sha256": checkpoint_sha,
        "safety": {"teacher_plant_actions": 0, "runtime_teacher_actions": 0,
                   "flight_sim_packets_sent": 0, "submission_authorized": False},
        "next_authority": (
            "One source-locked teacher-free residual-scale diagnostic ladder."
            if numerically_admitted else "Reject VG064 without rollout authority."
        ),
    }
    write_json_atomic(output / "report.json", report)
    state.update({"status": "completed" if numerically_admitted else "rejected",
                  "numerically_admitted": numerically_admitted,
                  "checkpoint_sha256": checkpoint_sha,
                  "report_sha256": sha256_path(output / "report.json")})
    write_json_atomic(state_path, state)
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
