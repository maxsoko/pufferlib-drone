#!/usr/bin/env python3
"""Train SF014 with simultaneous oracle-history and DAgger gradients."""

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
from scripts.train_vq2_recurrent_dagger import (
    BC_DATASET,
    BC_METADATA_SHA256,
    BC_REPORT_SHA256,
    DAGGER_DATASET,
    DAGGER_METADATA_SHA256,
    DAGGER_REPORT_SHA256,
    aggregate_validation_score,
)


TAG = "vq2_sf019_recurrent_dagger_paired_001"
SCHEMA = "vq2_recurrent_dagger_paired_checkpoint_v1"
DEFAULT_OUTPUT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG
)


@dataclass(frozen=True)
class PairedTrainConfig:
    seed: int = 42019
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


def paired_objective(
    bc_action_loss: torch.Tensor,
    bc_smoothness: torch.Tensor,
    dagger_action_loss: torch.Tensor,
    dagger_smoothness: torch.Tensor,
    *,
    smoothness_weight: float,
) -> torch.Tensor:
    """Equal source weight inside every optimizer update."""

    bc = bc_action_loss + smoothness_weight * bc_smoothness
    dagger = dagger_action_loss + smoothness_weight * dagger_smoothness
    return 0.5 * (bc + dagger)


class DaggerBatchStream:
    """Cycle complete DAgger episodes while preserving each recurrent prefix."""

    def __init__(
        self,
        dataset: OracleBCDataset,
        agents: np.ndarray,
        config: PairedTrainConfig,
        rng: np.random.Generator,
        device: torch.device,
    ) -> None:
        self.dataset = dataset
        self.agents = np.asarray(agents, dtype=np.int64)
        self.config = config
        self.rng = rng
        self.device = device
        self.order = np.empty(0, dtype=np.int64)
        self.batch_cursor = 0
        self.batch_agents = np.empty(0, dtype=np.int64)
        self.start = 0
        self.maximum = 0
        self.state: torch.Tensor | None = None
        self.previous_prediction: torch.Tensor | None = None
        self.previous_valid: torch.Tensor | None = None
        self.records = 0
        self.completed_episode_batches = 0
        self.completed_dataset_cycles = 0
        self._begin_next_batch()

    def _begin_next_batch(self) -> None:
        if self.batch_cursor >= len(self.order):
            self.order = self.rng.permutation(self.agents)
            self.batch_cursor = 0
            self.completed_dataset_cycles += 1
        end = min(
            self.batch_cursor + self.config.agent_batch_size, len(self.order)
        )
        self.batch_agents = self.order[self.batch_cursor:end]
        self.batch_cursor = end
        self.start = 0
        self.maximum = int(self.dataset.lengths[self.batch_agents].max())
        self.state = None
        self.previous_prediction = None
        self.previous_valid = None

    def next(
        self, model: VQ2RecurrentActor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if self.start >= self.maximum:
            self.completed_episode_batches += 1
            self._begin_next_batch()
        if self.state is None:
            self.state = model.initial_state(
                len(self.batch_agents), device=self.device
            )
        end = min(self.start + self.config.sequence_chunk, self.maximum)
        legal, target, valid = self.dataset.chunk(
            self.batch_agents, self.start, end, device=self.device
        )
        return (
            legal,
            target,
            valid,
            self.state,
            self.previous_prediction,
            self.previous_valid,
        )

    def commit(
        self,
        next_state: torch.Tensor,
        prediction: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        self.state = next_state.detach()
        self.previous_prediction = prediction[:, -1].detach()
        self.previous_valid = valid[:, -1].detach()
        self.start += prediction.shape[1]
        self.records += int(valid.sum().item())


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: PairedTrainConfig = PairedTrainConfig(),
    parent_checkpoint: Path = CHECKPOINT,
    parent_checkpoint_sha256: str = CHECKPOINT_SHA256,
    tag: str = TAG,
    preregistration: Path = (
        ROOT / "docs/vq2_sf019_recurrent_dagger_paired_preregistration_2026-07-28.md"
    ),
    extra_source_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF019 preregisters CUDA training")
    if sha256_path(parent_checkpoint) != parent_checkpoint_sha256:
        raise RuntimeError("paired parent checkpoint hash mismatch")
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
    train_agents, validation_agents = actor_agent_split(
        bc_dataset.agents, config.validation_agents
    )
    parent = torch.load(
        parent_checkpoint, map_location=device, weights_only=False
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

    parent_bc = evaluate(model, bc_dataset, validation_agents, config, device)
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
        model.train()
        bc_order = rng.permutation(train_agents)
        dagger_stream = DaggerBatchStream(
            dagger_dataset, train_agents, config, rng, device
        )
        bc_metrics = ActionAccumulator()
        dagger_metrics = ActionAccumulator()
        bc_loss_sum = 0.0
        dagger_loss_sum = 0.0
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
                    dagger_legal,
                    dagger_target,
                    dagger_valid,
                    dagger_state,
                    dagger_previous_prediction,
                    dagger_previous_valid,
                ) = dagger_stream.next(model)
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
                bc_state = bc_next_state.detach()
                bc_previous_prediction = bc_output.mean[:, -1].detach()
                bc_previous_valid = bc_valid[:, -1].detach()
                dagger_stream.commit(
                    dagger_next_state, dagger_output.mean, dagger_valid
                )
                bc_count = int(bc_valid.sum().item())
                dagger_count = int(dagger_valid.sum().item())
                bc_records += bc_count
                bc_loss_sum += float(bc_action_loss.detach().item()) * bc_count
                dagger_loss_sum += (
                    float(dagger_action_loss.detach().item()) * dagger_count
                )
                bc_metrics.add(bc_output.mean, bc_target, bc_valid)
                dagger_metrics.add(
                    dagger_output.mean, dagger_target, dagger_valid
                )

        bc_validation = evaluate(
            model, bc_dataset, validation_agents, config, device
        )
        dagger_validation = evaluate(
            model, dagger_dataset, validation_agents, config, device
        )
        score = aggregate_validation_score(bc_validation, dagger_validation)
        epoch_report = {
            "epoch": epoch,
            "bc_train": bc_metrics.result(),
            "dagger_train": dagger_metrics.result(),
            "bc_train_mean_loss": bc_loss_sum / max(bc_records, 1),
            "dagger_train_mean_loss": (
                dagger_loss_sum / max(dagger_stream.records, 1)
            ),
            "bc_train_records": bc_records,
            "dagger_train_records": dagger_stream.records,
            "dagger_completed_episode_batches": (
                dagger_stream.completed_episode_batches
            ),
            "dagger_completed_dataset_cycles": (
                dagger_stream.completed_dataset_cycles
            ),
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
        raise RuntimeError("paired training produced no selectable checkpoint")
    output.mkdir(parents=True)
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/train_vq2_recurrent_bc.py",
        ROOT / "scripts/train_vq2_recurrent_dagger.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_informed.py",
        preregistration,
        parent_checkpoint,
        BC_DATASET / "report.json",
        BC_DATASET / "metadata.json",
        DAGGER_DATASET / "report.json",
        DAGGER_DATASET / "metadata.json",
        *extra_source_paths,
    ]
    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    checkpoint = {
        "schema": SCHEMA,
        "tag": tag,
        "model": parent["model"],
        "model_state": best_state,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
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
    selected = history[best_epoch]
    report = {
        "schema": "vq2_recurrent_dagger_paired_training_report_v1",
        "tag": tag,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "best_epoch": best_epoch,
        "best_aggregate_validation_score": best_score,
        "best_bc_validation": selected["bc_validation"],
        "best_dagger_validation": selected["dagger_validation"],
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
