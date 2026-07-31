#!/usr/bin/env python3
"""Fit only a phase-gated Puffer action residual on warmed VG039 histories."""

from __future__ import annotations

import argparse
import json
import random
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
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.eval_vq2_variable_gate_oracle import write_json_once
from scripts.train_vq2_variable_gate_recurrent_bc import (
    VariableGateBCDataset,
    _agent_batches,
    actor_agent_split,
    atomic_torch_save,
    runtime_manifest,
    sha256_path,
)


TAG = "vq2_vg057_warmed_phase_residual_001"
REPORT_SCHEMA = "vq2_vg057_warmed_phase_residual_report_v1"
CHECKPOINT_SCHEMA = "vq2_vg057_warmed_phase_residual_checkpoint_v1"
STATE_SCHEMA = "vq2_vg057_warmed_phase_residual_state_v1"
PARENT_CHECKPOINT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg033_variable_gate_eight_source_refit_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = "56a8e3b84770c1a54afbf11ac2503478221f7829c28723da0f7dfcfa2a7aac26"
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = "033a7f1f5d1bfbe82c1cdd7b3d63407d99c220a899eb4420b259379728563f60"
PARENT_ADMISSION = ROOT / "docs/vq2_vg033_eight_source_refit_admission_2026-07-31.json"
PARENT_ADMISSION_SHA256 = "dbf0fbb41ebcca80f5284377db0c8b15d9be18a8880aa29a7a78513558617d44"
DATASET = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg039_variable_gate_dagger_round9_vg033_visited_512"
)
DATASET_REPORT_SHA256 = "a0bdcd0d60f66f7de55323a08b3c4695a0333b5bf88a93454d56d75e61eb21e1"
DATASET_METADATA_SHA256 = "89134752057b0dd7759031aaf524a93f0a85096bd1b8973ed6daebb5715c74ac"
DATASET_ADMISSION = ROOT / "docs/vq2_vg039_variable_gate_dagger_round9_admission_2026-07-31.json"
DATASET_ADMISSION_SHA256 = "e01eb2e28c65f827cfb39916c28d56320c57a4c23513ee61553447bd236b11cc"
REJECTION = ROOT / "docs/vq2_vg056_count6_paired_confirmation_rejection_2026-07-31.json"
REJECTION_SHA256 = "c97082e4f6b0d40c74ee841e9b0c42786194b3d004dc4bf2e49337282e6a87b1"
PACKAGING_FAILURE = ROOT / "docs/vq2_vg057_post_training_packaging_failure_2026-07-31.json"
PACKAGING_FAILURE_SHA256 = "627bca441ff024c27212f8d5c2d60d9f71a63eb10d22a84e53329aa446813b51"
GOAL = ROOT / "docs/vq2_48h_competitive_lap_goal_prompt_2026-07-31.md"
GOAL_SHA256 = "03f085d32a217889f56600ac2600bead24e087ee47322eb5aa2e208f231f2aa1"
PREREGISTRATION = ROOT / "docs/vq2_vg057_warmed_phase_residual_preregistration_2026-07-31.md"
RUNNER = ROOT / "scripts/run_vq2_vg057_vast.sh"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG

# Preserve early-course mass while making phase 4 (Gate 5 active) the target.
PHASE_ROW_WEIGHTS = (0.0, 0.25, 0.5, 2.0, 32.0, 32.0, 32.0, 32.0,
                     32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 429183
    hidden_size: int = 256
    initial_std: float = 0.15
    epochs: int = 6
    agent_batch_size: int = 8
    sequence_chunk: int = 256
    validation_agents: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    action_weights: tuple[float, float, float, float] = (1.0, 1.0, 4.0, 1.0)
    phase_row_weights: tuple[float, ...] = PHASE_ROW_WEIGHTS


def source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(), PREREGISTRATION, RUNNER, GOAL,
        PARENT_CHECKPOINT, PARENT_REPORT, PARENT_ADMISSION,
        DATASET / "report.json", DATASET / "metadata.json", DATASET_ADMISSION,
        REJECTION, PACKAGING_FAILURE,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        ROOT / "pufferlib/vq2_recurrent.py", ROOT / "pufferlib/vq2_informed.py",
        ROOT / "scripts/train_vq2_variable_gate_recurrent_bc.py",
    )


def source_identity() -> tuple[str, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return commit, {
        str(path.resolve().relative_to(ROOT)): sha256_path(path)
        for path in source_paths()
    }


def verify_inputs() -> None:
    expected = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        PARENT_ADMISSION: PARENT_ADMISSION_SHA256,
        DATASET / "report.json": DATASET_REPORT_SHA256,
        DATASET / "metadata.json": DATASET_METADATA_SHA256,
        DATASET_ADMISSION: DATASET_ADMISSION_SHA256,
        REJECTION: REJECTION_SHA256,
        PACKAGING_FAILURE: PACKAGING_FAILURE_SHA256,
        GOAL: GOAL_SHA256,
    }
    for path, digest in expected.items():
        if sha256_path(path) != digest:
            raise RuntimeError(f"VG057 source evidence changed: {path}")
    if not PREREGISTRATION.is_file() or not RUNNER.is_file():
        raise RuntimeError("VG057 source lock is incomplete")
    rejection = json.loads(REJECTION.read_text())
    packaging_failure = json.loads(PACKAGING_FAILURE.read_text())
    dataset = json.loads((DATASET / "report.json").read_text())
    if (
        rejection.get("schema") != "vq2_vg056_count6_paired_confirmation_rejection_v1"
        or not rejection.get("completed")
        or rejection.get("numerically_admitted")
        or not rejection.get("unchanged_retry_forbidden")
        or rejection.get("live_authority")
    ):
        raise RuntimeError("VG056 rejection does not authorize VG057")
    if (
        packaging_failure.get("schema")
        != "vq2_vg057_post_training_packaging_failure_v1"
        or packaging_failure.get("checkpoint_written")
        or packaging_failure.get("report_written")
        or packaging_failure.get("optimizer_updates") != 5330
        or packaging_failure.get("live_authority")
    ):
        raise RuntimeError("VG057 packaging repair evidence changed")
    if (
        dataset.get("schema") != "vq2_vg039_variable_gate_dagger_collection_report_v1"
        or not dataset.get("admitted")
        or dataset.get("records", 0) < 1_000_000
        or not dataset.get("admission_predicates", {}).get("minimum_phase_4_records")
        or dataset.get("safety", {}).get("flight_sim_packets_sent") != 0
    ):
        raise RuntimeError("VG039 warmed dataset is not admitted")


def load_model(device: torch.device) -> tuple[VQ2PhaseResidualActor, dict[str, Any]]:
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != "vq2_variable_gate_recurrent_bc_checkpoint_v1"
        or not payload.get("numerically_admitted")
        or contract.get("class") != "VQ2PhaseRecurrentActor"
        or contract.get("legal_observation_size") != PHASE_LEGAL_OBS_SIZE
        or contract.get("hidden_size") != 256
    ):
        raise RuntimeError("VG057 parent actor contract changed")
    model = VQ2PhaseResidualActor(hidden_size=256, initial_std=0.15).to(device)
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    if incompatible.unexpected_keys or incompatible.missing_keys != [
        "phase_action_residual.weight"
    ]:
        raise RuntimeError("VG057 residual is not the only new parameter")
    torch.nn.init.zeros_(model.phase_action_residual.weight)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.phase_action_residual.weight.requires_grad_(True)
    return model, payload


def base_exact(model: VQ2PhaseResidualActor, parent: dict[str, Any]) -> bool:
    state = model.state_dict()
    return all(torch.equal(state[name].cpu(), value.cpu()) for name, value in parent["model_state"].items())


def evaluate(
    model: VQ2PhaseResidualActor,
    dataset: VariableGateBCDataset,
    agents: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    sums = torch.zeros(17, dtype=torch.float64, device=device)
    counts = torch.zeros(17, dtype=torch.float64, device=device)
    action_weights = torch.tensor(config.action_weights, device=device)
    divisor = float(sum(config.action_weights))
    model.eval()
    with torch.no_grad():
        for batch_agents in _agent_batches(agents, config.agent_batch_size):
            state = model.initial_state(len(batch_agents), device=device)
            maximum = int(dataset.lengths[batch_agents].max())
            for start in range(0, maximum, config.sequence_chunk):
                end = min(start + config.sequence_chunk, maximum)
                observation, target, valid, _ = dataset.chunk(
                    batch_agents, start, end, device=device
                )
                output, state = model.forward_sequence(observation, state)
                phase = torch.round(observation[..., -1] * 16.0).long().clamp_(0, 16)
                error = ((output.mean - target).square() * action_weights).sum(-1) / divisor
                for index in range(17):
                    mask = valid & (phase == index)
                    if mask.any():
                        sums[index] += error[mask].double().sum()
                        counts[index] += mask.sum()
    return {
        "phase_counts": {str(i): int(counts[i].item()) for i in range(17) if counts[i]},
        "phase_weighted_mse": {
            str(i): float((sums[i] / counts[i]).item())
            for i in range(17) if counts[i]
        },
    }


def numerically_admitted(
    baseline: dict[str, Any],
    selected: dict[str, Any],
    *,
    base_parameters_exact: bool,
    residual_norm: float,
) -> bool:
    b, s = baseline["phase_weighted_mse"], selected["phase_weighted_mse"]
    return bool(
        base_parameters_exact
        and selected["phase_counts"].get("4", 0) >= 1000
        and s["4"] < b["4"]
        and s["3"] <= 1.02 * b["3"]
        and s["2"] <= 1.02 * b["2"]
        and np.isfinite(residual_norm)
        and residual_norm <= 8.0
    )


def train(
    *, output: Path = DEFAULT_OUTPUT, device_name: str = "cuda",
    config: TrainConfig = TrainConfig(), resume: bool = False,
) -> dict[str, Any]:
    if tuple(config.phase_row_weights) != PHASE_ROW_WEIGHTS:
        raise RuntimeError("VG057 phase weights changed")
    verify_inputs()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG057 requires CUDA")
    report_path, state_path = output / "report.json", output / "training_state.pt"
    commit, hashes = source_identity()
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        report = json.loads(report_path.read_text())
        if (
            report.get("source_commit") != commit
            or report.get("source_sha256") != hashes
            or sha256_path(output / "policy_best.pt")
            != report.get("checkpoint_sha256")
        ):
            raise RuntimeError("VG057 completed source changed")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if (
            state.get("status") != "completed"
            or state.get("source_commit") != commit
            or state.get("source_sha256") != hashes
            or state.get("checkpoint_sha256") != report.get("checkpoint_sha256")
            or state.get("report_sha256") != sha256_path(report_path)
        ):
            raise RuntimeError("VG057 completed state changed")
        return report
    if output.exists() and not state_path.is_file():
        raise RuntimeError("VG057 output exists without state")

    device = torch.device(device_name)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    dataset = VariableGateBCDataset(
        DATASET, expected_report_sha256=DATASET_REPORT_SHA256,
        expected_metadata_sha256=DATASET_METADATA_SHA256,
    )
    train_agents, validation_agents = actor_agent_split(dataset.agents, config.validation_agents)
    model, parent = load_model(device)
    trainable = [model.phase_action_residual.weight]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    baseline = evaluate(model, dataset, validation_agents, config, device)
    best_score = float(baseline["phase_weighted_mse"]["4"])
    best_admitted = False
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    completed_epoch = 0
    updates = 0
    identity = {
        "schema": STATE_SCHEMA, "tag": TAG, "source_commit": commit,
        "source_sha256": hashes, "runtime": runtime_manifest(),
        "train_config": asdict(config), "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_report_sha256": DATASET_REPORT_SHA256,
    }
    if state_path.is_file():
        if not resume: raise FileExistsError(f"VG057 state exists at {state_path}")
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        for key, value in identity.items():
            if saved.get(key) != value: raise RuntimeError(f"VG057 resume mismatch: {key}")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        baseline = saved["baseline_validation"]; best_score = saved["best_score"]
        best_admitted = bool(saved["best_admitted"])
        best_epoch = saved["best_epoch"]; best_state = saved["best_state"]
        history = saved["history"]; completed_epoch = saved["completed_epoch"]
        updates = saved["optimizer_updates"]
    else:
        if resume: raise RuntimeError("VG057 resume requested without state")
        output.mkdir(parents=True)
        atomic_torch_save(state_path, {**identity, "status": "training",
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "numpy_rng_state": rng.bit_generator.state, "baseline_validation": baseline,
            "best_score": best_score, "best_admitted": best_admitted,
            "best_epoch": best_epoch, "best_state": best_state,
            "history": history, "completed_epoch": 0, "optimizer_updates": 0})

    action_weights = torch.tensor(config.action_weights, device=device)
    row_lookup = torch.tensor(PHASE_ROW_WEIGHTS, device=device)
    divisor = float(sum(config.action_weights))
    started = time.perf_counter()
    for epoch in range(completed_epoch + 1, config.epochs + 1):
        model.train(); loss_sum = 0.0; weight_sum = 0.0
        for batch_agents in _agent_batches(rng.permutation(train_agents), config.agent_batch_size):
            state = model.initial_state(len(batch_agents), device=device)
            maximum = int(dataset.lengths[batch_agents].max())
            for start in range(0, maximum, config.sequence_chunk):
                end = min(start + config.sequence_chunk, maximum)
                observation, target, valid, _ = dataset.chunk(batch_agents, start, end, device=device)
                optimizer.zero_grad(set_to_none=True)
                actor_output, next_state = model.forward_sequence(observation, state)
                phase = torch.round(observation[..., -1] * 16.0).long().clamp_(0, 16)
                row_weight = row_lookup[phase] * valid
                if row_weight.sum() > 0:
                    error = ((actor_output.mean - target).square() * action_weights).sum(-1) / divisor
                    loss = (error * row_weight).sum() / row_weight.sum()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
                    optimizer.step(); updates += 1
                    weight = float(row_weight.sum().item())
                    loss_sum += float(loss.detach().item()) * weight; weight_sum += weight
                state = next_state.detach()
        validation = evaluate(model, dataset, validation_agents, config, device)
        score = float(validation["phase_weighted_mse"]["4"])
        residual_norm = float(model.phase_action_residual.weight.detach().norm().item())
        eligible = numerically_admitted(
            baseline, validation, base_parameters_exact=base_exact(model, parent),
            residual_norm=residual_norm,
        )
        item = {"epoch": epoch, "train_loss": loss_sum / max(weight_sum, 1.0),
                "validation": validation, "phase4_score": score,
                "residual_l2": residual_norm, "numerically_admitted": eligible,
                "optimizer_updates": updates}
        history.append(item); print(json.dumps(item, sort_keys=True), flush=True)
        if np.isfinite(score) and (eligible, -score) > (best_admitted, -best_score):
            best_score = score; best_admitted = eligible; best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        atomic_torch_save(state_path, {**identity, "status": "training",
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "numpy_rng_state": rng.bit_generator.state, "baseline_validation": baseline,
            "best_score": best_score, "best_admitted": best_admitted,
            "best_epoch": best_epoch, "best_state": best_state,
            "history": history, "completed_epoch": epoch, "optimizer_updates": updates})

    model.load_state_dict(best_state)
    selected = evaluate(model, dataset, validation_agents, config, device)
    residual_norm = float(model.phase_action_residual.weight.detach().norm().item())
    admitted = numerically_admitted(baseline, selected,
        base_parameters_exact=base_exact(model, parent), residual_norm=residual_norm)
    checkpoint = {**parent, "schema": CHECKPOINT_SCHEMA, "tag": TAG,
        "model": {**parent["model"], "class": "VQ2PhaseResidualActor"},
        "model_state": best_state, "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_report_sha256": DATASET_REPORT_SHA256, "train_config": asdict(config),
        "trainable_parameter_names": ["phase_action_residual.weight"],
        "trainable_parameters": 1024, "best_epoch": best_epoch,
        "optimizer_updates": updates, "numerically_admitted": admitted,
        "source_commit": commit, "source_sha256": hashes,
        "safety": {"actor_input_privileged_values": 0, "teacher_blend": 0.0,
                   "teacher_plant_actions": 0, "flight_sim_packets_sent": 0,
                   "submission_authorized": False}}
    checkpoint_path = output / "policy_best.pt"; atomic_torch_save(checkpoint_path, checkpoint)
    report = {"schema": REPORT_SCHEMA, "tag": TAG, "completed": True,
        "numerically_admitted": admitted, "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_report_sha256": DATASET_REPORT_SHA256, "best_epoch": best_epoch,
        "optimizer_updates": updates, "baseline_validation": baseline,
        "selected_validation": selected, "residual_l2": residual_norm,
        "base_parameters_exact": base_exact(model, parent), "history": history,
        "wall_time_seconds": time.perf_counter() - started,
        "source_commit": commit, "source_sha256": hashes, "safety": checkpoint["safety"],
        "next_authority": "Preregister a teacher-free six-gate residual-scale bracket." if admitted else "Reject this residual fit."}
    write_json_once(report_path, report)
    atomic_torch_save(state_path, {**identity, "status": "completed",
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "numpy_rng_state": rng.bit_generator.state, "baseline_validation": baseline,
        "best_score": best_score, "best_admitted": best_admitted,
        "best_epoch": best_epoch, "best_state": best_state,
        "history": history, "completed_epoch": config.epochs, "optimizer_updates": updates,
        "checkpoint_sha256": report["checkpoint_sha256"], "report_sha256": sha256_path(report_path)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device, resume=args.resume)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
