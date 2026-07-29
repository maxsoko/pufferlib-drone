#!/usr/bin/env python3
"""Fine-tune SF014 on a record-balanced SF012 + SF017 DAgger aggregate."""

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
from pufferlib.vq2_recurrent import ACTION_SIZE, VQ2RecurrentActor
from scripts.eval_vq2_recurrent_policy import CHECKPOINT, CHECKPOINT_SHA256
from scripts.train_vq2_recurrent_bc import (
    ActionAccumulator,
    OracleBCDataset,
    _agent_batches,
    actor_agent_split,
    evaluate,
    sha256_path,
    temporal_smoothness,
    weighted_action_mse,
)


TAG = "vq2_sf018_recurrent_dagger_001"
SCHEMA = "vq2_recurrent_dagger_checkpoint_v1"
DEFAULT_OUTPUT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG
)
BC_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf012_oracle_legal_bc_dataset_64"
)
DAGGER_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf017_recurrent_dagger_dataset_64"
)
BC_REPORT_SHA256 = (
    "9bb31d59d126b00a037566a7a5745af827037740f91c022a070e4b953de5746f"
)
BC_METADATA_SHA256 = (
    "21158fbcdcf749c54b459991e379ee315edf71ab2b0c1efaa3f73e38a7f2fbb2"
)
DAGGER_REPORT_SHA256 = (
    "58937d85d5cbd349a5bf7c2628f4d307c0e3ed97d68e85db850c2c49bcda687e"
)
DAGGER_METADATA_SHA256 = (
    "d50cbde39b6bd052b2ad2293b7a876bedf2c1aa8c4a7847261d3860eedefc39e"
)


@dataclass(frozen=True)
class DaggerTrainConfig:
    seed: int = 42018
    hidden_size: int = 256
    initial_std: float = 0.15
    epochs: int = 4
    agent_batch_size: int = 8
    sequence_chunk: int = 64
    validation_agents: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    smoothness_weight: float = 1e-4
    action_weights: tuple[float, float, float, float] = (1.0, 1.0, 4.0, 1.0)
    bc_passes_per_epoch: int = 1
    dagger_passes_per_epoch: int = 44


def source_schedule(config: DaggerTrainConfig, rng: np.random.Generator) -> list[str]:
    if config.bc_passes_per_epoch <= 0 or config.dagger_passes_per_epoch <= 0:
        raise ValueError("aggregate training requires both BC and DAgger passes")
    schedule = ["bc"] * config.bc_passes_per_epoch + [
        "dagger"
    ] * config.dagger_passes_per_epoch
    rng.shuffle(schedule)
    return schedule


def aggregate_validation_score(
    bc_validation: dict[str, Any], dagger_validation: dict[str, Any]
) -> float:
    bc = float(bc_validation["weighted_mse"])
    dagger = float(dagger_validation["weighted_mse"])
    if not np.isfinite(bc) or not np.isfinite(dagger):
        return float("inf")
    return 0.5 * (bc + dagger)


def train_source_pass(
    *,
    model: VQ2RecurrentActor,
    dataset: OracleBCDataset,
    train_agents: np.ndarray,
    config: DaggerTrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], float, int]:
    model.train()
    order = rng.permutation(train_agents)
    weights = torch.tensor(config.action_weights, device=device)
    metrics = ActionAccumulator()
    loss_sum = 0.0
    record_count = 0
    updates = 0
    for batch_agents in _agent_batches(order, config.agent_batch_size):
        state = model.initial_state(len(batch_agents), device=device)
        previous_prediction: torch.Tensor | None = None
        previous_valid: torch.Tensor | None = None
        maximum = int(dataset.lengths[batch_agents].max())
        for start in range(0, maximum, config.sequence_chunk):
            end = min(start + config.sequence_chunk, maximum)
            legal, target, valid = dataset.chunk(
                batch_agents, start, end, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                actor_output, next_state = model.forward_sequence(legal, state)
                action_loss, _ = weighted_action_mse(
                    actor_output.mean, target, valid, weights
                )
                smoothness = temporal_smoothness(
                    actor_output.mean,
                    valid,
                    previous_prediction=previous_prediction,
                    previous_valid=previous_valid,
                )
                loss = action_loss + config.smoothness_weight * smoothness
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            state = next_state.detach()
            previous_prediction = actor_output.mean[:, -1].detach()
            previous_valid = valid[:, -1].detach()
            count = int(valid.sum().item())
            record_count += count
            loss_sum += float(loss.detach().item()) * count
            metrics.add(actor_output.mean, target, valid)
    return metrics.result(), loss_sum / max(record_count, 1), updates


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: DaggerTrainConfig = DaggerTrainConfig(),
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF018 preregisters CUDA training")
    if sha256_path(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SF014 parent checkpoint hash mismatch")
    device = torch.device(device_name)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    bc_dataset = OracleBCDataset(
        BC_DATASET,
        verify_hashes=True,
        expected_report_sha256=BC_REPORT_SHA256,
        expected_metadata_sha256=BC_METADATA_SHA256,
    )
    dagger_dataset = OracleBCDataset(
        DAGGER_DATASET,
        verify_hashes=True,
        expected_report_sha256=DAGGER_REPORT_SHA256,
        expected_metadata_sha256=DAGGER_METADATA_SHA256,
    )
    if bc_dataset.agents != dagger_dataset.agents:
        raise RuntimeError("aggregate datasets do not share the course-agent ABI")
    train_agents, validation_agents = actor_agent_split(
        bc_dataset.agents, config.validation_agents
    )
    parent = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = VQ2RecurrentActor(
        hidden_size=config.hidden_size, initial_std=config.initial_std
    ).to(device)
    model.load_state_dict(parent["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    total_updates = 0
    started = time.perf_counter()

    parent_bc = evaluate(
        model, bc_dataset, validation_agents, config, device
    )
    parent_dagger = evaluate(
        model, dagger_dataset, validation_agents, config, device
    )
    history.append(
        {
            "epoch": 0,
            "parent": True,
            "bc_validation": parent_bc,
            "dagger_validation": parent_dagger,
            "aggregate_validation_score": aggregate_validation_score(
                parent_bc, parent_dagger
            ),
            "optimizer_updates": 0,
        }
    )

    for epoch in range(1, config.epochs + 1):
        schedule = source_schedule(config, rng)
        source_records = {"bc": 0, "dagger": 0}
        source_loss_sum = {"bc": 0.0, "dagger": 0.0}
        source_passes = {"bc": 0, "dagger": 0}
        for source in schedule:
            dataset = bc_dataset if source == "bc" else dagger_dataset
            metrics, loss, updates = train_source_pass(
                model=model,
                dataset=dataset,
                train_agents=train_agents,
                config=config,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                rng=rng,
            )
            records = int(metrics["count"])
            source_records[source] += records
            source_loss_sum[source] += loss * records
            source_passes[source] += 1
            total_updates += updates
        bc_validation = evaluate(
            model, bc_dataset, validation_agents, config, device
        )
        dagger_validation = evaluate(
            model, dagger_dataset, validation_agents, config, device
        )
        score = aggregate_validation_score(bc_validation, dagger_validation)
        epoch_report = {
            "epoch": epoch,
            "source_passes": source_passes,
            "source_records": source_records,
            "source_mean_loss": {
                source: source_loss_sum[source] / max(source_records[source], 1)
                for source in ("bc", "dagger")
            },
            "bc_validation": bc_validation,
            "dagger_validation": dagger_validation,
            "aggregate_validation_score": score,
            "optimizer_updates": total_updates,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("aggregate training produced no selectable checkpoint")
    output.mkdir(parents=True)
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/train_vq2_recurrent_bc.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "docs/vq2_sf018_recurrent_dagger_preregistration_2026-07-28.md",
        CHECKPOINT,
        BC_DATASET / "report.json",
        BC_DATASET / "metadata.json",
        DAGGER_DATASET / "report.json",
        DAGGER_DATASET / "metadata.json",
    ]
    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    checkpoint = {
        "schema": SCHEMA,
        "tag": TAG,
        "model": parent["model"],
        "model_state": best_state,
        "parent_checkpoint_sha256": CHECKPOINT_SHA256,
        "train_config": asdict(config),
        "train_agents": train_agents.tolist(),
        "validation_agents": validation_agents.tolist(),
        "bc_report_sha256": BC_REPORT_SHA256,
        "bc_metadata_sha256": BC_METADATA_SHA256,
        "dagger_report_sha256": DAGGER_REPORT_SHA256,
        "dagger_metadata_sha256": DAGGER_METADATA_SHA256,
        "best_epoch": best_epoch,
        "best_aggregate_validation_score": best_score,
        "history": history,
        "optimizer_updates": total_updates,
        "source_sha256": source_sha256,
        "safety": {
            "stored_privileged_values_per_actor_record": 0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }
    checkpoint_path = output / "policy_best.pt"
    torch.save(checkpoint, checkpoint_path)
    best_report = history[best_epoch]
    report = {
        "schema": "vq2_recurrent_dagger_training_report_v1",
        "tag": TAG,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": CHECKPOINT_SHA256,
        "best_epoch": best_epoch,
        "best_aggregate_validation_score": best_score,
        "best_bc_validation": best_report["bc_validation"],
        "best_dagger_validation": best_report["dagger_validation"],
        "history": history,
        "optimizer_updates": total_updates,
        "wall_time_seconds": time.perf_counter() - started,
        "source_sha256": source_sha256,
        "safety": checkpoint["safety"],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    report = train(output=args.output.resolve(), device_name=args.device)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
