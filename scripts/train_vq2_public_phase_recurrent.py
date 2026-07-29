#!/usr/bin/env python3
"""Fine-tune one phase-aware recurrent actor with a phase-zero anchor."""

from __future__ import annotations

import argparse
import json
import random
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
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_public_phase_adapter import (
    DATASET,
    DATASET_METADATA_SHA256,
    DATASET_REPORT_SHA256,
    PublicPhaseDataset,
    _agent_batches,
)
from scripts.train_vq2_recurrent_bc import (
    ActionAccumulator,
    actor_agent_split,
    temporal_smoothness,
    weighted_action_mse,
)


TAG = "vq2_sf042_public_phase_recurrent_fit_001"
SCHEMA = "vq2_public_phase_recurrent_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf042_public_phase_recurrent_fit_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf041_public_phase_residual_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "e31bbbe639fe9ab6727dd3f170c37504d6073bb4f001310a92d032f135872755"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "e7610e45c890a80a4fe1e0dc0ee8c905197a85e8870b9d789c5c8ff44c40e03b"
)


@dataclass(frozen=True)
class RecurrentPhaseConfig:
    seed: int = 42042
    hidden_size: int = 256
    initial_std: float = 0.15
    epochs: int = 8
    agent_batch_size: int = 8
    sequence_chunk: int = 64
    validation_agents: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    smoothness_weight: float = 1e-4
    phase_zero_loss_weight: float = 2.0
    gate2_loss_weight: float = 1.0
    train_encoder: bool = False
    action_weights: tuple[float, float, float, float] = (4.0, 4.0, 4.0, 1.0)


def phase_masks(
    observation: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = observation[..., -1]
    zero = torch.zeros((), device=phase.device, dtype=phase.dtype)
    gate2_value = torch.full((), 1.0 / 6.0, device=phase.device, dtype=phase.dtype)
    phase_zero = valid & torch.isclose(phase, zero, atol=1e-7, rtol=0.0)
    gate2 = valid & torch.isclose(phase, gate2_value, atol=1e-7, rtol=0.0)
    return phase_zero, gate2


def evaluate_partitions(
    model: VQ2PhaseResidualActor,
    dataset: PublicPhaseDataset,
    agents: np.ndarray,
    config: RecurrentPhaseConfig,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    model.eval()
    accumulators = {
        "phase_zero": ActionAccumulator(),
        "gate2": ActionAccumulator(),
    }
    weighted_sum = {name: 0.0 for name in accumulators}
    counts = {name: 0 for name in accumulators}
    weights = torch.tensor(config.action_weights, device=device)
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
                masks = dict(zip(accumulators, phase_masks(observation, valid)))
                for name, mask in masks.items():
                    if not bool(mask.any()):
                        continue
                    loss, _ = weighted_action_mse(
                        output.mean, target, mask, weights
                    )
                    count = int(mask.sum().item())
                    weighted_sum[name] += float(loss.item()) * count
                    counts[name] += count
                    accumulators[name].add(output.mean, target, mask)
    result: dict[str, dict[str, Any]] = {}
    for name, accumulator in accumulators.items():
        result[name] = accumulator.result()
        result[name]["weighted_mse"] = weighted_sum[name] / max(counts[name], 1)
    return result


def numerical_admission(partitions: dict[str, dict[str, Any]]) -> bool:
    phase_zero = partitions["phase_zero"]
    gate2 = partitions["gate2"]
    values = [
        float(phase_zero["weighted_mse"]),
        float(gate2["weighted_mse"]),
        *map(float, phase_zero["mse"]),
        *map(float, gate2["mse"]),
    ]
    return bool(
        np.isfinite(values).all()
        and values[0] <= 0.03
        and values[1] <= 0.02
        and all(value <= 0.05 for value in values[2:])
    )


def train(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: RecurrentPhaseConfig = RecurrentPhaseConfig(),
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF042 preregisters CUDA training")
    frozen = {
        PARENT_CHECKPOINT: PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: PARENT_REPORT_SHA256,
        DATASET / "report.json": DATASET_REPORT_SHA256,
        DATASET / "metadata.json": DATASET_METADATA_SHA256,
    }
    for path, expected in frozen.items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"source-lock mismatch for {path}")
    device = torch.device(device_name)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    dataset = PublicPhaseDataset()
    train_agents, validation_agents = actor_agent_split(
        dataset.agents, config.validation_agents
    )
    parent = torch.load(PARENT_CHECKPOINT, map_location=device, weights_only=False)
    if parent.get("schema") not in {
        "vq2_public_phase_residual_checkpoint_v1",
        "vq2_public_phase_recurrent_checkpoint_v1",
    }:
        raise RuntimeError("SF041 parent schema changed")
    model = VQ2PhaseResidualActor(
        hidden_size=config.hidden_size, initial_std=config.initial_std
    ).to(device)
    model.load_state_dict(parent["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.phase_embedding,
        model.recurrent,
        model.action_head,
        model.phase_action_residual,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if config.train_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(True)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(name == "log_std" for name in trainable_names) or (
        not config.train_encoder
        and any(name.startswith("encoder.") for name in trainable_names)
    ):
        raise RuntimeError("phase recurrent trainable boundary changed")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = torch.tensor(config.action_weights, device=device)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []
    best_rank = (1, float("inf"))
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    updates = 0
    started = time.perf_counter()

    parent_validation = evaluate_partitions(
        model, dataset, validation_agents, config, device
    )
    history.append(
        {
            "epoch": 0,
            "parent": True,
            "validation": parent_validation,
            "numerically_admitted": numerical_admission(parent_validation),
            "optimizer_updates": 0,
        }
    )
    for epoch in range(1, config.epochs + 1):
        model.train()
        metrics = {
            "phase_zero": ActionAccumulator(),
            "gate2": ActionAccumulator(),
        }
        loss_sum = 0.0
        update_records = 0
        for batch_agents in _agent_batches(
            rng.permutation(train_agents), config.agent_batch_size
        ):
            state = model.initial_state(len(batch_agents), device=device)
            previous_prediction: torch.Tensor | None = None
            previous_selected: torch.Tensor | None = None
            maximum = int(dataset.lengths[batch_agents].max())
            for start in range(0, maximum, config.sequence_chunk):
                end = min(start + config.sequence_chunk, maximum)
                observation, target, valid, _ = dataset.chunk(
                    batch_agents, start, end, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    output, next_state = model.forward_sequence(observation, state)
                    phase_zero, gate2 = phase_masks(observation, valid)
                    terms: list[torch.Tensor] = []
                    if bool(phase_zero.any()):
                        zero_loss, _ = weighted_action_mse(
                            output.mean, target, phase_zero, weights
                        )
                        terms.append(config.phase_zero_loss_weight * zero_loss)
                    if bool(gate2.any()):
                        gate2_loss, _ = weighted_action_mse(
                            output.mean, target, gate2, weights
                        )
                        terms.append(config.gate2_loss_weight * gate2_loss)
                    selected = phase_zero | gate2
                    smoothness = temporal_smoothness(
                        output.mean,
                        selected,
                        previous_prediction=previous_prediction,
                        previous_valid=previous_selected,
                    )
                    loss = sum(terms) + config.smoothness_weight * smoothness
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                updates += 1
                count = int(selected.sum().item())
                loss_sum += float(loss.detach().item()) * count
                update_records += count
                metrics["phase_zero"].add(output.mean, target, phase_zero)
                metrics["gate2"].add(output.mean, target, gate2)
                state = next_state.detach()
                previous_prediction = output.mean[:, -1].detach()
                previous_selected = selected[:, -1].detach()
        validation = evaluate_partitions(
            model, dataset, validation_agents, config, device
        )
        admitted = numerical_admission(validation)
        epoch_report = {
            "epoch": epoch,
            "train_loss": loss_sum / max(update_records, 1),
            "train": {name: value.result() for name, value in metrics.items()},
            "validation": validation,
            "numerically_admitted": admitted,
            "optimizer_updates": updates,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        score = sum(float(value["weighted_mse"]) for value in validation.values())
        rank = (0 if admitted else 1, score)
        if np.isfinite(score) and rank < best_rank:
            best_rank = rank
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("SF042 produced no finite checkpoint")

    output_path.mkdir(parents=True)
    source_paths = [
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "scripts/train_vq2_public_phase_adapter.py",
        PARENT_CHECKPOINT,
        PARENT_REPORT,
        DATASET / "report.json",
        DATASET / "metadata.json",
    ]
    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    selected = history[best_epoch]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    checkpoint = {
        "schema": SCHEMA,
        "tag": TAG,
        "model": parent["model"],
        "model_state": best_state,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_report_sha256": DATASET_REPORT_SHA256,
        "dataset_metadata_sha256": DATASET_METADATA_SHA256,
        "train_config": asdict(config),
        "trainable_parameter_names": trainable_names,
        "trainable_parameters": trainable_count,
        "train_agents": train_agents.tolist(),
        "validation_agents": validation_agents.tolist(),
        "best_epoch": best_epoch,
        "best_validation": selected["validation"],
        "numerically_admitted": selected["numerically_admitted"],
        "history": history,
        "optimizer_updates": updates,
        "source_sha256": source_sha256,
        "safety": {
            "actor_input_public_status_values": 1,
            "stored_training_only_privileged_values_per_actor_record": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    checkpoint_path = output_path / "policy_best.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": "vq2_public_phase_recurrent_training_report_v1",
        "tag": TAG,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_report_sha256": DATASET_REPORT_SHA256,
        "dataset_metadata_sha256": DATASET_METADATA_SHA256,
        "best_epoch": best_epoch,
        "best_validation": selected["validation"],
        "parent_validation": parent_validation,
        "numerically_admitted": selected["numerically_admitted"],
        "trainable_parameter_names": trainable_names,
        "trainable_parameters": trainable_count,
        "history": history,
        "optimizer_updates": updates,
        "wall_time_seconds": time.perf_counter() - started,
        "source_sha256": source_sha256,
        "safety": checkpoint["safety"],
    }
    (output_path / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    report = train(output_path=args.output.resolve(), device_name=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
