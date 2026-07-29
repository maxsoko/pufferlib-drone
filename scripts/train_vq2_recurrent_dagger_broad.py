#!/usr/bin/env python3
"""Continue SF019 on simultaneous SF012 and broad SF021 DAgger gradients."""

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
    aggregate_validation_score,
)
from scripts.train_vq2_recurrent_dagger_paired import (
    DaggerBatchStream,
    paired_objective,
)


TAG = "vq2_sf022_recurrent_dagger_broad_001"
SCHEMA = "vq2_recurrent_dagger_broad_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf022_recurrent_dagger_broad_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf019_recurrent_dagger_paired_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "a391516076f49549255bf323f94859f90a0d15a5f3406c436d3045682e681352"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "9311411d3ab30321623575b904fb018e33396d88b856f7e0554a764926984c4c"
)
BROAD_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf021_recurrent_dagger_broad_512"
)
BROAD_REPORT_SHA256 = (
    "438cc4ba435a1cba223fba204fbc2637802ce68c193e5994c5e56e076f1f6c4b"
)
BROAD_METADATA_SHA256 = (
    "ae315f9469112e069c3b68b72e848c7faa56c21cb79a2dbd4bd4a7a4c2273875"
)


@dataclass(frozen=True)
class BroadTrainConfig:
    seed: int = 42022
    hidden_size: int = 256
    initial_std: float = 0.15
    epochs: int = 4
    agent_batch_size: int = 8
    sequence_chunk: int = 64
    bc_validation_agents: int = 8
    broad_validation_agents: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    smoothness_weight: float = 1e-4
    action_weights: tuple[float, float, float, float] = (1.0, 1.0, 4.0, 1.0)


def broad_training_admitted(
    bc_validation: dict[str, Any], broad_validation: dict[str, Any]
) -> bool:
    values = [
        float(bc_validation["weighted_mse"]),
        float(broad_validation["weighted_mse"]),
        *[float(value) for value in broad_validation["mse"]],
    ]
    return (
        bool(np.isfinite(values).all())
        and values[0] <= 0.01
        and values[1] <= 0.02
        and all(value <= 0.05 for value in values[2:])
    )


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: BroadTrainConfig = BroadTrainConfig(),
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF022 preregisters CUDA training")
    if sha256_path(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("SF019 parent checkpoint hash mismatch")
    if sha256_path(PARENT_REPORT) != PARENT_REPORT_SHA256:
        raise RuntimeError("SF019 parent report hash mismatch")

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
    bc_train_agents, bc_validation_agents = actor_agent_split(
        bc_dataset.agents, config.bc_validation_agents
    )
    broad_train_agents, broad_validation_agents = actor_agent_split(
        broad_dataset.agents, config.broad_validation_agents
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
    weights = torch.tensor(config.action_weights, device=device)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    total_updates = 0
    started = time.perf_counter()

    parent_bc = evaluate(
        model, bc_dataset, bc_validation_agents, config, device
    )
    parent_broad = evaluate(
        model, broad_dataset, broad_validation_agents, config, device
    )
    history.append(
        {
            "epoch": 0,
            "parent": True,
            "bc_validation": parent_bc,
            "broad_validation": parent_broad,
            "aggregate_validation_score": aggregate_validation_score(
                parent_bc, parent_broad
            ),
            "numerically_admitted": broad_training_admitted(
                parent_bc, parent_broad
            ),
            "optimizer_updates": 0,
        }
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        bc_order = rng.permutation(bc_train_agents)
        broad_stream = DaggerBatchStream(
            broad_dataset, broad_train_agents, config, rng, device
        )
        bc_metrics = ActionAccumulator()
        broad_metrics = ActionAccumulator()
        bc_loss_sum = 0.0
        broad_loss_sum = 0.0
        bc_records = 0
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
                (
                    broad_legal,
                    broad_target,
                    broad_valid,
                    broad_state,
                    broad_previous_prediction,
                    broad_previous_valid,
                ) = broad_stream.next(model)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    bc_output, bc_next_state = model.forward_sequence(
                        bc_legal, bc_state
                    )
                    broad_output, broad_next_state = model.forward_sequence(
                        broad_legal, broad_state
                    )
                    bc_action_loss, _ = weighted_action_mse(
                        bc_output.mean, bc_target, bc_valid, weights
                    )
                    broad_action_loss, _ = weighted_action_mse(
                        broad_output.mean, broad_target, broad_valid, weights
                    )
                    bc_smoothness = temporal_smoothness(
                        bc_output.mean,
                        bc_valid,
                        previous_prediction=bc_previous_prediction,
                        previous_valid=bc_previous_valid,
                    )
                    broad_smoothness = temporal_smoothness(
                        broad_output.mean,
                        broad_valid,
                        previous_prediction=broad_previous_prediction,
                        previous_valid=broad_previous_valid,
                    )
                    loss = paired_objective(
                        bc_action_loss,
                        bc_smoothness,
                        broad_action_loss,
                        broad_smoothness,
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
                bc_state = bc_next_state.detach()
                bc_previous_prediction = bc_output.mean[:, -1].detach()
                bc_previous_valid = bc_valid[:, -1].detach()
                broad_stream.commit(
                    broad_next_state, broad_output.mean, broad_valid
                )
                bc_count = int(bc_valid.sum().item())
                broad_count = int(broad_valid.sum().item())
                bc_records += bc_count
                bc_loss_sum += float(bc_action_loss.detach().item()) * bc_count
                broad_loss_sum += (
                    float(broad_action_loss.detach().item()) * broad_count
                )
                bc_metrics.add(bc_output.mean, bc_target, bc_valid)
                broad_metrics.add(
                    broad_output.mean, broad_target, broad_valid
                )

        bc_validation = evaluate(
            model, bc_dataset, bc_validation_agents, config, device
        )
        broad_validation = evaluate(
            model, broad_dataset, broad_validation_agents, config, device
        )
        score = aggregate_validation_score(bc_validation, broad_validation)
        epoch_report = {
            "epoch": epoch,
            "bc_train": bc_metrics.result(),
            "broad_train": broad_metrics.result(),
            "bc_train_mean_loss": bc_loss_sum / max(bc_records, 1),
            "broad_train_mean_loss": (
                broad_loss_sum / max(broad_stream.records, 1)
            ),
            "bc_train_records": bc_records,
            "broad_train_records": broad_stream.records,
            "broad_completed_episode_batches": (
                broad_stream.completed_episode_batches
            ),
            "broad_completed_dataset_cycles": (
                broad_stream.completed_dataset_cycles
            ),
            "bc_validation": bc_validation,
            "broad_validation": broad_validation,
            "aggregate_validation_score": score,
            "numerically_admitted": broad_training_admitted(
                bc_validation, broad_validation
            ),
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
        raise RuntimeError("broad paired training produced no checkpoint")
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
        "bc_train_agents": bc_train_agents.tolist(),
        "bc_validation_agents": bc_validation_agents.tolist(),
        "broad_train_agents": broad_train_agents.tolist(),
        "broad_validation_agents": broad_validation_agents.tolist(),
        "bc_report_sha256": BC_REPORT_SHA256,
        "bc_metadata_sha256": BC_METADATA_SHA256,
        "broad_report_sha256": BROAD_REPORT_SHA256,
        "broad_metadata_sha256": BROAD_METADATA_SHA256,
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
        "schema": "vq2_recurrent_dagger_broad_training_report_v1",
        "tag": TAG,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "best_epoch": best_epoch,
        "best_aggregate_validation_score": best_score,
        "best_bc_validation": selected["bc_validation"],
        "best_broad_validation": selected["broad_validation"],
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
