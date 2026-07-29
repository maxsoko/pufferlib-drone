#!/usr/bin/env python3
"""Fit the phase-gated joint residual on SF039 Gate-2 states."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE, VQ2RecurrentActor
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_public_phase_adapter import (
    DATASET,
    DATASET_METADATA_SHA256,
    DATASET_REPORT_SHA256,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    PARENT_REPORT,
    PARENT_REPORT_SHA256,
    PhaseAdapterConfig,
    PublicPhaseDataset,
    _agent_batches,
    evaluate_gate2,
    numerical_admission,
    phase_zero_is_exact,
)
from scripts.train_vq2_recurrent_bc import (
    ActionAccumulator,
    actor_agent_split,
    temporal_smoothness,
    weighted_action_mse,
)


TAG = "vq2_sf041_public_phase_residual_001"
SCHEMA = "vq2_public_phase_residual_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf041_public_phase_residual_preregistration_2026-07-28.md"
)
CONFIG = replace(PhaseAdapterConfig(), seed=42041)


def train(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: PhaseAdapterConfig = CONFIG,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF041 preregisters CUDA training")
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
    payload = torch.load(PARENT_CHECKPOINT, map_location=device, weights_only=False)
    parent = VQ2RecurrentActor(
        hidden_size=config.hidden_size, initial_std=config.initial_std
    ).to(device)
    parent.load_state_dict(payload["model_state"])
    parent.eval()
    model = VQ2PhaseResidualActor(
        hidden_size=config.hidden_size, initial_std=config.initial_std
    ).to(device)
    model.load_phase_zero_base_state(payload["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.phase_embedding.weight.requires_grad_(True)
    model.phase_action_residual.weight.requires_grad_(True)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count != 5 * config.hidden_size:
        raise RuntimeError("SF041 must train exactly 1,280 phase-path parameters")
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

    parent_validation = evaluate_gate2(
        model, dataset, validation_agents, config, device
    )
    history.append(
        {
            "epoch": 0,
            "parent": True,
            "gate2_validation": parent_validation,
            "phase_zero_exact": phase_zero_is_exact(
                model, parent, dataset, validation_agents, device
            ),
            "numerically_admitted": False,
            "optimizer_updates": 0,
        }
    )
    for epoch in range(1, config.epochs + 1):
        model.train()
        metrics = ActionAccumulator()
        loss_sum = 0.0
        count_sum = 0
        for batch_agents in _agent_batches(
            rng.permutation(train_agents), config.agent_batch_size
        ):
            state = model.initial_state(len(batch_agents), device=device)
            previous_prediction: torch.Tensor | None = None
            previous_gate2: torch.Tensor | None = None
            maximum = int(dataset.lengths[batch_agents].max())
            for start in range(0, maximum, config.sequence_chunk):
                end = min(start + config.sequence_chunk, maximum)
                observation, target, _, gate2 = dataset.chunk(
                    batch_agents, start, end, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    actor_output, next_state = model.forward_sequence(
                        observation, state
                    )
                    if bool(gate2.any()):
                        action_loss, _ = weighted_action_mse(
                            actor_output.mean, target, gate2, weights
                        )
                        smoothness = temporal_smoothness(
                            actor_output.mean,
                            gate2,
                            previous_prediction=previous_prediction,
                            previous_valid=previous_gate2,
                        )
                        loss = action_loss + config.smoothness_weight * smoothness
                if bool(gate2.any()):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    updates += 1
                    count = int(gate2.sum().item())
                    loss_sum += float(loss.detach().item()) * count
                    count_sum += count
                    metrics.add(actor_output.mean, target, gate2)
                state = next_state.detach()
                previous_prediction = actor_output.mean[:, -1].detach()
                previous_gate2 = gate2[:, -1].detach()
        validation = evaluate_gate2(
            model, dataset, validation_agents, config, device
        )
        zero_exact = phase_zero_is_exact(
            model, parent, dataset, validation_agents, device
        )
        admitted = numerical_admission(validation, zero_exact)
        epoch_report = {
            "epoch": epoch,
            "train_loss": loss_sum / max(count_sum, 1),
            "gate2_train": metrics.result(),
            "gate2_validation": validation,
            "phase_zero_exact": zero_exact,
            "numerically_admitted": admitted,
            "optimizer_updates": updates,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        score = float(validation["weighted_mse"])
        rank = (0 if admitted else 1, score)
        if np.isfinite(score) and rank < best_rank:
            best_rank = rank
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("SF041 produced no finite checkpoint")

    output_path.mkdir(parents=True)
    source_paths = [
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "pufferlib/vq2_recurrent_phase_residual.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
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
    checkpoint = {
        "schema": SCHEMA,
        "tag": TAG,
        "model": {
            "class": "VQ2PhaseResidualActor",
            "legal_observation_size": PHASE_LEGAL_OBS_SIZE,
            "legacy_legal_observation_size": LEGAL_OBS_SIZE,
            "public_status_values": 1,
            "action_size": ACTION_SIZE,
            "hidden_size": config.hidden_size,
            "initial_std": config.initial_std,
        },
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
        "best_gate2_validation": selected["gate2_validation"],
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
        "schema": "vq2_public_phase_residual_training_report_v1",
        "tag": TAG,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "dataset_report_sha256": DATASET_REPORT_SHA256,
        "dataset_metadata_sha256": DATASET_METADATA_SHA256,
        "best_epoch": best_epoch,
        "best_gate2_validation": selected["gate2_validation"],
        "parent_gate2_validation": parent_validation,
        "phase_zero_exact": selected["phase_zero_exact"],
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

