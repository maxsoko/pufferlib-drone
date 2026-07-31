#!/usr/bin/env python3
"""Replay VG068 while retaining each independent residual head at its best epoch."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseMLPResidualActor
from scripts.collect_vq2_vg062_warmed_teacher_intervention_features import FEATURE_DTYPE
from scripts.eval_vq2_variable_gate_oracle import sha256_path, write_json_atomic
from scripts.train_vq2_variable_gate_recurrent_bc import atomic_torch_save
import scripts.train_vq2_vg068_indexed_mlp_intervention_features as vg068


TAG = "vq2_vg069_phase_independent_early_stop_001"
SCHEMA = "vq2_vg069_phase_independent_early_stop_report_v1"
STATE_SCHEMA = "vq2_vg069_phase_independent_early_stop_state_v1"
CHECKPOINT_SCHEMA = "vq2_vg069_phase_independent_early_stop_checkpoint_v1"
CONFIG = vg068.CONFIG
TARGET_PHASES = vg068.TARGET_PHASES
VG068 = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg068_indexed_mlp_intervention_features_001"
)
VG068_REPORT = VG068 / "report.json"
VG068_REPORT_SHA256 = "d2f8e3cadf75537760c4310ddb849baa57ba239705fbfe1a30fce98ccc8bcb46"
VG068_REJECTION = ROOT / "docs/vq2_vg068_indexed_mlp_intervention_rejection_2026-07-31.json"
VG068_REJECTION_SHA256 = "1f8273e954f68f828306423b6e5ec567470c7abcbd65c0323d3f53af83327b18"
PREREGISTRATION = ROOT / "docs/vq2_vg069_phase_independent_early_stop_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg069_vast.sh"
TEST = ROOT / "tests/test_train_vq2_vg069_phase_independent_early_stop.py"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PHASE_PARAMETER_NAMES = (
    "indexed_phase_residual_input",
    "indexed_phase_residual_input_bias",
    "indexed_phase_residual_output",
    "indexed_phase_residual_output_bias",
)
NEXT_AUTHORITY_ADMITTED = (
    "One source-locked teacher-free nonlinear residual scale diagnostic."
)


def checkpoint_model_contract(
    parent: dict[str, Any], config: Any
) -> dict[str, Any]:
    return {
        **parent["model"],
        "class": "VQ2IndexedPhaseMLPResidualActor",
        "residual_size": config.residual_size,
    }


def numerical_admission(
    metrics: dict[str, Any], *, base_exact: bool, non_target_zero: bool,
    trainable_l2: float, config: Any,
) -> bool:
    return vg068.numerically_admitted(
        metrics,
        base_exact=base_exact,
        non_target_zero=non_target_zero,
        trainable_l2=trainable_l2,
        config=config,
    )


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, TEST,
        VG068_REPORT, VG068_REJECTION, *vg068.source_paths(),
    )


def verify_inputs() -> dict[str, Any]:
    vg068.verify_inputs()
    if sha256_path(VG068_REPORT) != VG068_REPORT_SHA256:
        raise RuntimeError("VG069 bound VG068 report changed")
    if sha256_path(VG068_REJECTION) != VG068_REJECTION_SHA256:
        raise RuntimeError("VG069 bound VG068 rejection changed")
    report = json.loads(VG068_REPORT.read_text())
    rejection = json.loads(VG068_REJECTION.read_text())
    if (
        report.get("schema") != vg068.SCHEMA
        or report.get("numerically_admitted")
        or report.get("best_epoch") != 7
        or report.get("checkpoint_sha256")
        != "4f8b3d342d3c11d75d8893d8a9782759c45cc2bf42614df3617b48243513da53"
    ):
        raise RuntimeError("VG069 requires the exact rejected VG068 fit")
    successor = rejection.get("authorized_successor", {})
    if (
        rejection.get("schema") != "vq2_vg068_indexed_mlp_intervention_rejection_v1"
        or not rejection.get("terminally_rejected")
        or not rejection.get("unchanged_retry_forbidden")
        or successor.get("tag") != TAG
        or successor.get("rollout_authority")
        or rejection.get("live_authority")
    ):
        raise RuntimeError("VG068 does not authorize VG069")
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


def capture_phase(
    model: VQ2IndexedPhaseMLPResidualActor, phase: int,
) -> dict[str, torch.Tensor]:
    return {
        name: getattr(model, name)[phase].detach().clone()
        for name in PHASE_PARAMETER_NAMES
    }


def restore_phase(
    model: VQ2IndexedPhaseMLPResidualActor,
    phase: int,
    values: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name in PHASE_PARAMETER_NAMES:
            getattr(model, name)[phase].copy_(values[name])


def update_phase_bests(
    model: VQ2IndexedPhaseMLPResidualActor,
    metrics: dict[str, Any],
    epoch: int,
    best_objectives: dict[int, float],
    best_epochs: dict[int, int],
    best_states: dict[int, dict[str, torch.Tensor]],
) -> None:
    for phase in TARGET_PHASES:
        objective = float(metrics["phases"][str(phase)]["selected_action_mse"])
        if objective < best_objectives[phase]:
            best_objectives[phase] = objective
            best_epochs[phase] = epoch
            best_states[phase] = capture_phase(model, phase)


def replay_metric_error(
    history: list[dict[str, Any]], vg068_report: dict[str, Any],
) -> float:
    maximum = 0.0
    expected = vg068_report["history"]
    if len(history) != len(expected):
        return math.inf
    for actual_epoch, expected_epoch in zip(history, expected, strict=True):
        if actual_epoch["epoch"] != expected_epoch["epoch"]:
            return math.inf
        actual = actual_epoch["validation"]
        reference = expected_epoch["validation"]
        maximum = max(
            maximum,
            abs(
                actual["phase_balanced_action_mse"]
                - reference["phase_balanced_action_mse"]
            ),
        )
        for phase in TARGET_PHASES:
            maximum = max(
                maximum,
                abs(
                    actual["phases"][str(phase)]["selected_action_mse"]
                    - reference["phases"][str(phase)]["selected_action_mse"]
                ),
            )
    return maximum


def fit(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG069 preregisters CUDA training")
    report_path = output / "report.json"
    state_path = output.with_name(f"{output.name}_state.json")
    training_state_path = output.with_name(f"{output.name}_training_state.pt")
    if report_path.is_file():
        if not resume:
            raise RuntimeError("VG069 terminal output exists; use --resume")
        return json.loads(report_path.read_text())
    started = time.perf_counter()
    vg068_report = verify_inputs()
    commit, hashes = source_identity()
    device = torch.device(device_name)
    model, parent = vg068.load_model(device, CONFIG)
    records = np.memmap(vg068.FEATURES, mode="r", dtype=FEATURE_DTYPE)
    baseline = vg068.validation_metrics(model, records, device, CONFIG)
    optimizer = torch.optim.AdamW(
        vg068.trainable_parameters(model), lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )
    chunks = math.ceil(records.shape[0] / CONFIG.chunk_rows)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_objectives = {
        phase: baseline["phases"][str(phase)]["baseline_action_mse"]
        for phase in TARGET_PHASES
    }
    best_epochs = {phase: 0 for phase in TARGET_PHASES}
    best_states = {phase: capture_phase(model, phase) for phase in TARGET_PHASES}
    state_identity = {
        "schema": STATE_SCHEMA, "tag": TAG,
        "source_commit": commit, "source_sha256": hashes,
        "runtime": runtime_manifest(), "train_config": asdict(CONFIG),
        "feature_records": int(records.shape[0]),
        "feature_sha256": vg068.FEATURES_SHA256,
        "vg068_report_sha256": VG068_REPORT_SHA256,
        "safety": {"teacher_plant_actions": 0, "runtime_teacher_actions": 0,
                   "flight_sim_packets_sent": 0, "submission_authorized": False},
    }
    if state_path.is_file():
        if not resume or not training_state_path.is_file():
            raise RuntimeError("VG069 state exists without authorized resume")
        state = json.loads(state_path.read_text())
        for key, value in state_identity.items():
            if state.get(key) != value:
                raise RuntimeError(f"VG069 resume mismatch: {key}")
        payload = torch.load(training_state_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        history = payload["history"]
        start_epoch = int(payload["completed_epoch"]) + 1
        best_objectives = payload["best_objectives"]
        best_epochs = payload["best_epochs"]
        best_states = payload["best_states"]
    else:
        state = {**state_identity, "status": "running", "completed_epoch": 0}
        write_json_atomic(state_path, state)

    for epoch in range(start_epoch, CONFIG.epochs + 1):
        model.train()
        order = np.random.default_rng(CONFIG.seed + epoch).permutation(chunks)
        epoch_loss = 0.0
        updates = 0
        for chunk_index in order:
            start = int(chunk_index) * CONFIG.chunk_rows
            chunk = records[start : min(start + CONFIG.chunk_rows, records.shape[0])]
            held = vg068.validation_mask(chunk["agent_index"], CONFIG)
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
                prediction = vg068.phase_action_prediction(model, hidden, base, phase)
                phase_losses.append(torch.square(prediction - teacher).mean())
            if not phase_losses:
                continue
            loss = torch.stack(phase_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("VG069 emitted non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                vg068.trainable_parameters(model), CONFIG.gradient_clip
            )
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            updates += 1
        metrics = vg068.validation_metrics(model, records, device, CONFIG)
        update_phase_bests(
            model, metrics, epoch, best_objectives, best_epochs, best_states
        )
        history.append({
            "epoch": epoch, "optimizer_updates": updates,
            "mean_training_loss": epoch_loss / max(updates, 1),
            "validation": metrics,
        })
        atomic_torch_save(training_state_path, {
            "completed_epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "history": history,
            "best_objectives": best_objectives, "best_epochs": best_epochs,
            "best_states": best_states,
        })
        state.update({"completed_epoch": epoch,
                      "best_epochs": {str(k): v for k, v in best_epochs.items()}})
        write_json_atomic(state_path, state)

    composed, _ = vg068.load_model(device, CONFIG)
    for phase in TARGET_PHASES:
        restore_phase(composed, phase, best_states[phase])
    selected = vg068.validation_metrics(composed, records, device, CONFIG)
    replay_error = replay_metric_error(history, vg068_report)
    trainable_l2 = float(torch.sqrt(sum(
        torch.square(parameter.detach()).sum()
        for parameter in vg068.trainable_parameters(composed)
    )).item())
    base_exact = vg068.base_parameters_exact(composed, parent)
    non_target_zero = vg068.non_target_outputs_zero(composed)
    admitted = bool(
        replay_error <= 1e-12
        and numerical_admission(
            selected, base_exact=base_exact, non_target_zero=non_target_zero,
            trainable_l2=trainable_l2, config=CONFIG,
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        **parent, "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "model": checkpoint_model_contract(parent, CONFIG),
        "model_state": {
            name: value.detach().cpu() for name, value in composed.state_dict().items()
        },
        "train_config": {**asdict(CONFIG), "selection": "independent_phase_early_stop"},
        "best_phase_epochs": {str(k): v for k, v in best_epochs.items()},
        "numerically_admitted": admitted, "source_commit": commit,
        "source_sha256": hashes, "vg068_report_sha256": VG068_REPORT_SHA256,
        "selected_validation": selected,
        "safety": {"actor_input_privileged_values": 0, "runtime_teacher_actions": 0,
                   "flight_sim_packets_sent": 0, "submission_authorized": False},
    }
    atomic_torch_save(output / "policy_best.pt", checkpoint)
    checkpoint_sha = sha256_path(output / "policy_best.pt")
    report = {
        "schema": SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted, "source_commit": commit,
        "source_sha256": hashes, "runtime": runtime_manifest(),
        "train_config": asdict(CONFIG), "wall_time_seconds": time.perf_counter() - started,
        "feature_records": int(records.shape[0]), "feature_sha256": vg068.FEATURES_SHA256,
        "vg068_report_sha256": VG068_REPORT_SHA256,
        "vg068_history_replay_max_abs_error": replay_error,
        "baseline_validation": baseline, "selected_validation": selected,
        "history": history, "best_phase_epochs": {str(k): v for k, v in best_epochs.items()},
        "base_parameters_exact": base_exact,
        "non_target_outputs_zero": non_target_zero,
        "trainable_l2": trainable_l2, "checkpoint": "policy_best.pt",
        "checkpoint_sha256": checkpoint_sha,
        "safety": {"teacher_plant_actions": 0, "runtime_teacher_actions": 0,
                   "flight_sim_packets_sent": 0, "submission_authorized": False},
        "next_authority": (
            NEXT_AUTHORITY_ADMITTED
            if admitted else "Reject VG069 without rollout authority."
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
