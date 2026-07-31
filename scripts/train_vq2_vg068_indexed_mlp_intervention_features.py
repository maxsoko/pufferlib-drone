#!/usr/bin/env python3
"""Fit nonlinear public-indexed Puffer residuals on VG063 warmed features."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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
from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseMLPResidualActor
from scripts.collect_vq2_vg062_warmed_teacher_intervention_features import (
    FEATURE_DTYPE,
    HIDDEN_SIZE,
)
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_atomic
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save


TAG = "vq2_vg068_indexed_mlp_intervention_features_001"
SCHEMA = "vq2_vg068_indexed_mlp_intervention_report_v1"
STATE_SCHEMA = "vq2_vg068_indexed_mlp_intervention_state_v1"
CHECKPOINT_SCHEMA = "vq2_vg068_indexed_mlp_intervention_checkpoint_v1"
TARGET_PHASES = tuple(range(1, 12))
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
LINE_REJECTION = ROOT / "docs/vq2_vg067_sparse_early_head_count5_bracket_rejection_2026-07-31.json"
LINE_REJECTION_SHA256 = "12b1fc130b464439d3bdd7d5cb4fa632dbb84c957711574752c4a6bc128d1159"
GOAL = ROOT / "docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"
GOAL_SHA256 = "03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1"
PREREGISTRATION = ROOT / "docs/vq2_vg068_indexed_mlp_intervention_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg068_vast.sh"
TEST = ROOT / "tests/test_train_vq2_vg068_indexed_mlp_intervention_features.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 429203
    epochs: int = 10
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    residual_size: int = 64
    chunk_rows: int = 65536
    validation_agent_group_size: int = 8
    validation_agent_modulus: int = 8
    validation_agent_remainder: int = 0
    minimum_aggregate_improvement: float = 2.0
    minimum_early_phase_improvement: float = 2.0
    maximum_phase_regression_factor: float = 1.0
    maximum_trainable_l2: float = 256.0


CONFIG = TrainConfig()


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST, GOAL,
        PARENT_CHECKPOINT, PARENT_REPORT, DATASET_REPORT, FEATURES,
        DATASET_ADMISSION, LINE_REJECTION,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
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
        LINE_REJECTION: LINE_REJECTION_SHA256,
        GOAL: GOAL_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG068 bound input changed: {path}")
    report = json.loads(DATASET_REPORT.read_text())
    admission = json.loads(DATASET_ADMISSION.read_text())
    rejection = json.loads(LINE_REJECTION.read_text())
    if (
        not report.get("training_dataset_admitted")
        or report.get("metrics", {}).get("env/success_rate") != 1.0
        or report.get("metrics", {}).get("env/crash") != 0.0
        or report.get("feature_sha256") != FEATURES_SHA256
        or report.get("feature_records")
        != FEATURES.stat().st_size // FEATURE_DTYPE.itemsize
    ):
        raise RuntimeError("VG068 feature corpus is not admitted VG063")
    if (
        not admission.get("training_dataset_admitted")
        or admission.get("dataset", {}).get("feature_sha256") != FEATURES_SHA256
        or admission.get("live_authority")
    ):
        raise RuntimeError("VG068 dataset admission changed")
    if (
        rejection.get("schema")
        != "vq2_vg067_sparse_early_head_count5_bracket_rejection_v1"
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("live_authority")
    ):
        raise RuntimeError("VG067 does not authorize the nonlinear successor")
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


def validation_mask(
    agent_index: np.ndarray, config: TrainConfig = CONFIG,
) -> np.ndarray:
    values = np.asarray(agent_index, dtype=np.int64)
    return (
        (values // config.validation_agent_group_size)
        % config.validation_agent_modulus
        == config.validation_agent_remainder
    )


def load_model(
    device: torch.device, config: TrainConfig = CONFIG,
) -> tuple[VQ2IndexedPhaseMLPResidualActor, dict[str, Any]]:
    parent = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = parent.get("model", {})
    if (
        parent.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or not parent.get("numerically_admitted")
        or contract.get("class") != "VQ2PhaseRecurrentActor"
        or contract.get("hidden_size") != HIDDEN_SIZE
        or contract.get("legal_observation_size") != PHASE_LEGAL_OBS_SIZE
    ):
        raise RuntimeError("VG068 parent actor changed")
    torch.manual_seed(config.seed)
    model = VQ2IndexedPhaseMLPResidualActor(
        hidden_size=HIDDEN_SIZE,
        residual_size=config.residual_size,
        initial_std=float(contract["initial_std"]),
    ).to(device)
    model.load_base_state(parent["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in trainable_parameters(model):
        parameter.requires_grad_(True)
    return model, parent


def trainable_parameters(
    model: VQ2IndexedPhaseMLPResidualActor,
) -> list[torch.nn.Parameter]:
    return [
        model.indexed_phase_residual_input,
        model.indexed_phase_residual_input_bias,
        model.indexed_phase_residual_output,
        model.indexed_phase_residual_output_bias,
    ]


def phase_action_prediction(
    model: VQ2IndexedPhaseMLPResidualActor,
    hidden: torch.Tensor,
    base_pre_tanh: torch.Tensor,
    phase: int,
) -> torch.Tensor:
    feature = torch.tanh(
        hidden @ model.indexed_phase_residual_input[phase].T
        + model.indexed_phase_residual_input_bias[phase]
    )
    residual = (
        feature @ model.indexed_phase_residual_output[phase].T
        + model.indexed_phase_residual_output_bias[phase]
    )
    return torch.tanh(base_pre_tanh + residual)


def base_parameters_exact(
    model: VQ2IndexedPhaseMLPResidualActor, parent: dict[str, Any],
) -> bool:
    state = model.state_dict()
    return all(
        torch.equal(state[name].detach().cpu(), value.detach().cpu())
        for name, value in parent["model_state"].items()
    )


def non_target_outputs_zero(model: VQ2IndexedPhaseMLPResidualActor) -> bool:
    phases = (0, *range(12, 17))
    return bool(
        all(
            torch.count_nonzero(model.indexed_phase_residual_output[phase]) == 0
            and torch.count_nonzero(model.indexed_phase_residual_output_bias[phase]) == 0
            for phase in phases
        )
    )


def validation_metrics(
    model: VQ2IndexedPhaseMLPResidualActor,
    records: np.memmap,
    device: torch.device,
    config: TrainConfig = CONFIG,
) -> dict[str, Any]:
    selected_sse = {phase: 0.0 for phase in TARGET_PHASES}
    baseline_sse = {phase: 0.0 for phase in TARGET_PHASES}
    counts = {phase: 0 for phase in TARGET_PHASES}
    model.eval()
    with torch.no_grad():
        for start in range(0, records.shape[0], config.chunk_rows):
            chunk = records[start : min(start + config.chunk_rows, records.shape[0])]
            held = validation_mask(chunk["agent_index"], config)
            phases = np.asarray(chunk["phase_index"], dtype=np.int64)
            for phase in TARGET_PHASES:
                indices = np.flatnonzero(held & (phases == phase))
                if not indices.size:
                    continue
                hidden = torch.from_numpy(
                    np.array(chunk["hidden"][indices], dtype=np.float32, copy=True)
                ).to(device)
                base = torch.from_numpy(
                    np.array(chunk["base_pre_tanh"][indices], dtype=np.float32, copy=True)
                ).to(device)
                teacher = torch.from_numpy(
                    np.array(chunk["teacher_action"][indices], dtype=np.float32, copy=True)
                ).to(device)
                prediction = phase_action_prediction(model, hidden, base, phase)
                selected_sse[phase] += float(torch.square(prediction - teacher).sum().item())
                baseline_sse[phase] += float(torch.square(torch.tanh(base) - teacher).sum().item())
                counts[phase] += int(indices.size)
    phases: dict[str, Any] = {}
    for phase in TARGET_PHASES:
        if counts[phase] == 0:
            raise RuntimeError(f"VG068 phase {phase} has no validation records")
        denominator = counts[phase] * ACTION_SIZE
        baseline = baseline_sse[phase] / denominator
        selected = selected_sse[phase] / denominator
        phases[str(phase)] = {
            "records": counts[phase],
            "baseline_action_mse": baseline,
            "selected_action_mse": selected,
            "improvement_factor": baseline / max(selected, 1e-20),
        }
    return {
        "phases": phases,
        "phase_balanced_action_mse": float(
            np.mean([item["selected_action_mse"] for item in phases.values()])
        ),
        "baseline_phase_balanced_action_mse": float(
            np.mean([item["baseline_action_mse"] for item in phases.values()])
        ),
    }


def numerically_admitted(
    metrics: dict[str, Any], *, base_exact: bool, non_target_zero: bool,
    trainable_l2: float, config: TrainConfig = CONFIG,
) -> bool:
    phases = metrics["phases"]
    aggregate_factor = (
        metrics["baseline_phase_balanced_action_mse"]
        / max(metrics["phase_balanced_action_mse"], 1e-20)
    )
    return bool(
        base_exact and non_target_zero
        and math.isfinite(trainable_l2)
        and trainable_l2 <= config.maximum_trainable_l2
        and aggregate_factor >= config.minimum_aggregate_improvement
        and all(
            phases[str(phase)]["improvement_factor"]
            >= config.minimum_early_phase_improvement
            for phase in (1, 2, 3)
        )
        and all(
            item["selected_action_mse"]
            <= item["baseline_action_mse"] * config.maximum_phase_regression_factor
            for item in phases.values()
        )
    )


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
    resume: bool = False, config: TrainConfig = CONFIG,
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG068 preregisters CUDA training")
    report_path = output / "report.json"
    state_path = output.with_name(f"{output.name}_state.json")
    training_state_path = output.with_name(f"{output.name}_training_state.pt")
    if report_path.is_file():
        if not resume:
            raise RuntimeError("VG068 terminal output exists; use --resume")
        return json.loads(report_path.read_text())
    started = time.perf_counter()
    dataset_report = verify_inputs()
    commit, hashes = source_identity()
    device = torch.device(device_name)
    model, parent = load_model(device, config)
    records = np.memmap(FEATURES, mode="r", dtype=FEATURE_DTYPE)
    # Measure the frozen parent before any resumable child state is restored.
    # This keeps admission identical for uninterrupted and resumed fits.
    baseline = validation_metrics(model, records, device, config)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    chunks = math.ceil(records.shape[0] / config.chunk_rows)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_objective = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    state_identity = {
        "schema": STATE_SCHEMA, "tag": TAG,
        "source_commit": commit, "source_sha256": hashes,
        "runtime": runtime_manifest(), "train_config": asdict(config),
        "feature_records": int(records.shape[0]),
        "feature_sha256": FEATURES_SHA256,
        "safety": {"teacher_plant_actions": 0, "runtime_teacher_actions": 0,
                   "flight_sim_packets_sent": 0, "submission_authorized": False},
    }
    if state_path.is_file():
        if not resume or not training_state_path.is_file():
            raise RuntimeError("VG068 state exists without authorized resume")
        state = json.loads(state_path.read_text())
        for key, value in state_identity.items():
            if state.get(key) != value:
                raise RuntimeError(f"VG068 resume mismatch: {key}")
        payload = torch.load(training_state_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        history = payload["history"]
        start_epoch = int(payload["completed_epoch"]) + 1
        best_objective = float(payload["best_objective"])
        best_epoch = int(payload["best_epoch"])
        best_state = payload["best_state"]
    else:
        state = {**state_identity, "status": "running", "completed_epoch": 0}
        write_json_atomic(state_path, state)
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        order = np.random.default_rng(config.seed + epoch).permutation(chunks)
        epoch_loss = 0.0
        updates = 0
        for chunk_index in order:
            start = int(chunk_index) * config.chunk_rows
            chunk = records[start : min(start + config.chunk_rows, records.shape[0])]
            held = validation_mask(chunk["agent_index"], config)
            phases = np.asarray(chunk["phase_index"], dtype=np.int64)
            optimizer.zero_grad(set_to_none=True)
            phase_losses: list[torch.Tensor] = []
            for phase in TARGET_PHASES:
                indices = np.flatnonzero((~held) & (phases == phase))
                if not indices.size:
                    continue
                hidden = torch.from_numpy(
                    np.array(chunk["hidden"][indices], dtype=np.float32, copy=True)
                ).to(device)
                base = torch.from_numpy(
                    np.array(chunk["base_pre_tanh"][indices], dtype=np.float32, copy=True)
                ).to(device)
                teacher = torch.from_numpy(
                    np.array(chunk["teacher_action"][indices], dtype=np.float32, copy=True)
                ).to(device)
                prediction = phase_action_prediction(model, hidden, base, phase)
                phase_losses.append(torch.square(prediction - teacher).mean())
            if not phase_losses:
                continue
            loss = torch.stack(phase_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("VG068 emitted non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters(model), config.gradient_clip
            )
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            updates += 1
        metrics = validation_metrics(model, records, device, config)
        objective = metrics["phase_balanced_action_mse"]
        if objective < best_objective:
            best_objective = objective
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "epoch": epoch, "optimizer_updates": updates,
            "mean_training_loss": epoch_loss / max(updates, 1),
            "validation": metrics,
        })
        training_state = {
            "completed_epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_objective": best_objective,
            "best_epoch": best_epoch,
            "best_state": best_state,
        }
        atomic_torch_save(training_state_path, training_state)
        state.update({"completed_epoch": epoch, "best_epoch": best_epoch,
                      "best_objective": best_objective})
        write_json_atomic(state_path, state)

    model.load_state_dict(best_state)
    selected = validation_metrics(model, records, device, config)
    # Baseline is invariant; retain its exact first-pass values in every phase.
    for phase in TARGET_PHASES:
        selected["phases"][str(phase)]["baseline_action_mse"] = (
            baseline["phases"][str(phase)]["baseline_action_mse"]
        )
        selected["phases"][str(phase)]["improvement_factor"] = (
            selected["phases"][str(phase)]["baseline_action_mse"]
            / max(selected["phases"][str(phase)]["selected_action_mse"], 1e-20)
        )
    selected["baseline_phase_balanced_action_mse"] = (
        baseline["baseline_phase_balanced_action_mse"]
    )
    trainable_l2 = float(torch.sqrt(sum(
        torch.square(parameter.detach()).sum()
        for parameter in trainable_parameters(model)
    )).item())
    base_exact = base_parameters_exact(model, parent)
    non_target_zero = non_target_outputs_zero(model)
    admitted = numerically_admitted(
        selected, base_exact=base_exact, non_target_zero=non_target_zero,
        trainable_l2=trainable_l2, config=config,
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        **parent,
        "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "model": {**parent["model"], "class": "VQ2IndexedPhaseMLPResidualActor",
                  "residual_size": config.residual_size},
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "train_config": asdict(config), "best_epoch": best_epoch,
        "numerically_admitted": admitted,
        "source_commit": commit, "source_sha256": hashes,
        "dataset_report_sha256": DATASET_REPORT_SHA256,
        "dataset_feature_sha256": FEATURES_SHA256,
        "dataset_admission_sha256": DATASET_ADMISSION_SHA256,
        "selected_validation": selected,
        "safety": {"actor_input_privileged_values": 0, "runtime_teacher_actions": 0,
                   "flight_sim_packets_sent": 0, "submission_authorized": False},
    }
    atomic_torch_save(output / "policy_best.pt", checkpoint)
    checkpoint_sha = sha256_path(output / "policy_best.pt")
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted,
        "source_commit": commit, "source_sha256": hashes,
        "runtime": runtime_manifest(), "train_config": asdict(config),
        "wall_time_seconds": time.perf_counter() - started,
        "feature_records": int(records.shape[0]), "feature_sha256": FEATURES_SHA256,
        "dataset_success_rate": dataset_report["metrics"]["env/success_rate"],
        "baseline_validation": baseline, "selected_validation": selected,
        "history": history, "best_epoch": best_epoch,
        "base_parameters_exact": base_exact,
        "non_target_outputs_zero": non_target_zero,
        "trainable_l2": trainable_l2,
        "checkpoint": "policy_best.pt", "checkpoint_sha256": checkpoint_sha,
        "safety": {"teacher_plant_actions": 0, "runtime_teacher_actions": 0,
                   "flight_sim_packets_sent": 0, "submission_authorized": False},
        "next_authority": (
            "One source-locked teacher-free nonlinear residual scale diagnostic."
            if admitted else "Reject VG068 without rollout authority."
        ),
    }
    write_json_atomic(report_path, report)
    state.update({"status": "completed" if admitted else "rejected",
                  "numerically_admitted": admitted,
                  "checkpoint_sha256": checkpoint_sha,
                  "report_sha256": sha256_path(report_path)})
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
