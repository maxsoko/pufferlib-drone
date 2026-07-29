#!/usr/bin/env python3
"""Train SF022 on a record-balanced aggregate of two DAgger iterations."""

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

from pufferlib.vq2_recurrent import VQ2RecurrentActor
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
from scripts.train_vq2_recurrent_dagger import (
    BC_DATASET,
    BC_METADATA_SHA256,
    BC_REPORT_SHA256,
)
from scripts.train_vq2_recurrent_dagger_broad import (
    BROAD_DATASET,
    BROAD_METADATA_SHA256,
    BROAD_REPORT_SHA256,
)
from scripts.train_vq2_recurrent_dagger_paired import (
    DaggerBatchStream,
    paired_objective,
)


TAG = "vq2_sf025_recurrent_dagger_aggregate_001"
SCHEMA = "vq2_recurrent_dagger_aggregate_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf025_recurrent_dagger_aggregate_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf022_recurrent_dagger_broad_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "57949ba391ffdc4d1e0dda5aa352fbb7b6dcb2a51b1f7742018926f6c6b2d903"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "2cfc253ef13dc4a438a784df595a52947aaa0bcdb42320a6069bebdfa85f2f66"
)
NEXT_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf024_recurrent_dagger_next_512"
)
NEXT_REPORT_SHA256 = (
    "5f3d344643a75a66cdbb1b159ee2105c563201774d8e7122d19e888fc414cb26"
)
NEXT_METADATA_SHA256 = (
    "1b7491bc8130ff1cb1a2a6b4c64cf7abb580723ccaa65600f842df3af8a498fa"
)
DAGGER_SOURCE_PATTERN = ("broad", "next", "next", "next")


@dataclass(frozen=True)
class AggregateTrainConfig:
    seed: int = 42025
    hidden_size: int = 256
    initial_std: float = 0.15
    epochs: int = 4
    agent_batch_size: int = 8
    sequence_chunk: int = 64
    bc_validation_agents: int = 8
    dagger_validation_agents: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    smoothness_weight: float = 1e-4
    action_weights: tuple[float, float, float, float] = (1.0, 1.0, 4.0, 1.0)
    loss_action_weights: tuple[float, float, float, float] | None = None
    prefer_admitted: bool = False


def aggregate_validation_score(
    bc: dict[str, Any], broad: dict[str, Any], next_: dict[str, Any]
) -> float:
    dagger_count = float(broad["count"] + next_["count"])
    dagger = (
        float(broad["weighted_mse"]) * float(broad["count"])
        + float(next_["weighted_mse"]) * float(next_["count"])
    ) / dagger_count
    values = [float(bc["weighted_mse"]), dagger]
    if not np.isfinite(values).all():
        return float("inf")
    return 0.5 * (values[0] + values[1])


def aggregate_training_admitted(
    bc: dict[str, Any], broad: dict[str, Any], next_: dict[str, Any]
) -> bool:
    values = [
        float(bc["weighted_mse"]),
        float(broad["weighted_mse"]),
        float(next_["weighted_mse"]),
        *[float(value) for value in broad["mse"]],
        *[float(value) for value in next_["mse"]],
    ]
    return (
        bool(np.isfinite(values).all())
        and values[0] <= 0.01
        and values[1] <= 0.02
        and values[2] <= 0.02
        and all(value <= 0.05 for value in values[3:])
    )


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: AggregateTrainConfig = AggregateTrainConfig(),
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF025 preregisters CUDA training")
    if sha256_path(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("SF022 parent checkpoint hash mismatch")
    if sha256_path(PARENT_REPORT) != PARENT_REPORT_SHA256:
        raise RuntimeError("SF022 parent report hash mismatch")

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
    broad_dataset = OracleBCDataset(
        BROAD_DATASET,
        verify_hashes=True,
        expected_report_sha256=BROAD_REPORT_SHA256,
        expected_metadata_sha256=BROAD_METADATA_SHA256,
    )
    next_dataset = OracleBCDataset(
        NEXT_DATASET,
        verify_hashes=True,
        expected_report_sha256=NEXT_REPORT_SHA256,
        expected_metadata_sha256=NEXT_METADATA_SHA256,
    )
    bc_train, bc_validation_agents = actor_agent_split(
        bc_dataset.agents, config.bc_validation_agents
    )
    broad_train, broad_validation_agents = actor_agent_split(
        broad_dataset.agents, config.dagger_validation_agents
    )
    next_train, next_validation_agents = actor_agent_split(
        next_dataset.agents, config.dagger_validation_agents
    )

    parent = torch.load(
        PARENT_CHECKPOINT, map_location=device, weights_only=False
    )
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
    loss_action_weights = (
        config.loss_action_weights
        if config.loss_action_weights is not None
        else config.action_weights
    )
    weights = torch.tensor(loss_action_weights, device=device)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_selection_rank = (1, float("inf"))
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    total_updates = 0
    started = time.perf_counter()

    def validation() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return (
            evaluate(model, bc_dataset, bc_validation_agents, config, device),
            evaluate(
                model, broad_dataset, broad_validation_agents, config, device
            ),
            evaluate(model, next_dataset, next_validation_agents, config, device),
        )

    parent_bc, parent_broad, parent_next = validation()
    history.append(
        {
            "epoch": 0,
            "parent": True,
            "bc_validation": parent_bc,
            "broad_validation": parent_broad,
            "next_validation": parent_next,
            "aggregate_validation_score": aggregate_validation_score(
                parent_bc, parent_broad, parent_next
            ),
            "numerically_admitted": aggregate_training_admitted(
                parent_bc, parent_broad, parent_next
            ),
            "optimizer_updates": 0,
        }
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        bc_order = rng.permutation(bc_train)
        streams = {
            "broad": DaggerBatchStream(
                broad_dataset, broad_train, config, rng, device
            ),
            "next": DaggerBatchStream(
                next_dataset, next_train, config, rng, device
            ),
        }
        bc_metrics = ActionAccumulator()
        dagger_metrics = {
            "broad": ActionAccumulator(),
            "next": ActionAccumulator(),
        }
        bc_loss_sum = 0.0
        dagger_loss_sum = {"broad": 0.0, "next": 0.0}
        bc_records = 0
        epoch_update = 0
        for bc_agents in _agent_batches(bc_order, config.agent_batch_size):
            bc_state = model.initial_state(len(bc_agents), device=device)
            bc_previous_prediction: torch.Tensor | None = None
            bc_previous_valid: torch.Tensor | None = None
            bc_maximum = int(bc_dataset.lengths[bc_agents].max())
            for start in range(0, bc_maximum, config.sequence_chunk):
                end = min(start + config.sequence_chunk, bc_maximum)
                bc_legal, bc_target, bc_valid = bc_dataset.chunk(
                    bc_agents, start, end, device=device
                )
                source = DAGGER_SOURCE_PATTERN[
                    epoch_update % len(DAGGER_SOURCE_PATTERN)
                ]
                stream = streams[source]
                (
                    dagger_legal,
                    dagger_target,
                    dagger_valid,
                    dagger_state,
                    dagger_previous_prediction,
                    dagger_previous_valid,
                ) = stream.next(model)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    bc_output, bc_next_state = model.forward_sequence(
                        bc_legal, bc_state
                    )
                    dagger_output, dagger_next_state = model.forward_sequence(
                        dagger_legal, dagger_state
                    )
                    bc_action_loss, _ = weighted_action_mse(
                        bc_output.mean, bc_target, bc_valid, weights
                    )
                    dagger_action_loss, _ = weighted_action_mse(
                        dagger_output.mean, dagger_target, dagger_valid, weights
                    )
                    bc_smoothness = temporal_smoothness(
                        bc_output.mean,
                        bc_valid,
                        previous_prediction=bc_previous_prediction,
                        previous_valid=bc_previous_valid,
                    )
                    dagger_smoothness = temporal_smoothness(
                        dagger_output.mean,
                        dagger_valid,
                        previous_prediction=dagger_previous_prediction,
                        previous_valid=dagger_previous_valid,
                    )
                    loss = paired_objective(
                        bc_action_loss,
                        bc_smoothness,
                        dagger_action_loss,
                        dagger_smoothness,
                        smoothness_weight=config.smoothness_weight,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
                total_updates += 1
                epoch_update += 1
                bc_state = bc_next_state.detach()
                bc_previous_prediction = bc_output.mean[:, -1].detach()
                bc_previous_valid = bc_valid[:, -1].detach()
                stream.commit(dagger_next_state, dagger_output.mean, dagger_valid)
                bc_count = int(bc_valid.sum().item())
                dagger_count = int(dagger_valid.sum().item())
                bc_records += bc_count
                bc_loss_sum += float(bc_action_loss.detach().item()) * bc_count
                dagger_loss_sum[source] += (
                    float(dagger_action_loss.detach().item()) * dagger_count
                )
                bc_metrics.add(bc_output.mean, bc_target, bc_valid)
                dagger_metrics[source].add(
                    dagger_output.mean, dagger_target, dagger_valid
                )

        bc_val, broad_val, next_val = validation()
        score = aggregate_validation_score(bc_val, broad_val, next_val)
        epoch_report = {
            "epoch": epoch,
            "bc_train": bc_metrics.result(),
            "broad_train": dagger_metrics["broad"].result(),
            "next_train": dagger_metrics["next"].result(),
            "bc_train_mean_loss": bc_loss_sum / max(bc_records, 1),
            "broad_train_mean_loss": (
                dagger_loss_sum["broad"] / max(streams["broad"].records, 1)
            ),
            "next_train_mean_loss": (
                dagger_loss_sum["next"] / max(streams["next"].records, 1)
            ),
            "bc_train_records": bc_records,
            "broad_train_records": streams["broad"].records,
            "next_train_records": streams["next"].records,
            "broad_completed_dataset_cycles": (
                streams["broad"].completed_dataset_cycles
            ),
            "next_completed_dataset_cycles": (
                streams["next"].completed_dataset_cycles
            ),
            "bc_validation": bc_val,
            "broad_validation": broad_val,
            "next_validation": next_val,
            "aggregate_validation_score": score,
            "numerically_admitted": aggregate_training_admitted(
                bc_val, broad_val, next_val
            ),
            "optimizer_updates": total_updates,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        admitted = bool(epoch_report["numerically_admitted"])
        selection_rank = (
            (0 if admitted else 1, score)
            if config.prefer_admitted
            else (0, score)
        )
        if selection_rank < best_selection_rank:
            best_selection_rank = selection_rank
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("aggregate training produced no checkpoint")
    output.mkdir(parents=True)
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/train_vq2_recurrent_dagger_paired.py",
        ROOT / "scripts/train_vq2_recurrent_bc.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_informed.py",
        PREREGISTRATION,
        PARENT_CHECKPOINT,
        PARENT_REPORT,
        BC_DATASET / "report.json",
        BC_DATASET / "metadata.json",
        BROAD_DATASET / "report.json",
        BROAD_DATASET / "metadata.json",
        NEXT_DATASET / "report.json",
        NEXT_DATASET / "metadata.json",
    ]
    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    checkpoint = {
        "schema": SCHEMA,
        "tag": TAG,
        "model": parent["model"],
        "model_state": best_state,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "train_config": asdict(config),
        "dagger_source_pattern": list(DAGGER_SOURCE_PATTERN),
        "bc_train_agents": bc_train.tolist(),
        "bc_validation_agents": bc_validation_agents.tolist(),
        "broad_train_agents": broad_train.tolist(),
        "broad_validation_agents": broad_validation_agents.tolist(),
        "next_train_agents": next_train.tolist(),
        "next_validation_agents": next_validation_agents.tolist(),
        "dataset_sha256": {
            "bc_report": BC_REPORT_SHA256,
            "bc_metadata": BC_METADATA_SHA256,
            "broad_report": BROAD_REPORT_SHA256,
            "broad_metadata": BROAD_METADATA_SHA256,
            "next_report": NEXT_REPORT_SHA256,
            "next_metadata": NEXT_METADATA_SHA256,
        },
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
    selected = history[best_epoch]
    report = {
        "schema": "vq2_recurrent_dagger_aggregate_training_report_v1",
        "tag": TAG,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "best_epoch": best_epoch,
        "best_aggregate_validation_score": best_score,
        "best_bc_validation": selected["bc_validation"],
        "best_broad_validation": selected["broad_validation"],
        "best_next_validation": selected["next_validation"],
        "numerically_admitted": selected["numerically_admitted"],
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
