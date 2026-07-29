#!/usr/bin/env python3
"""Train the smallest full-history recurrent Puffer actor on SF012."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE, VQ2RecurrentActor


TAG = "vq2_sf014_recurrent_bc_001"
SCHEMA = "vq2_recurrent_bc_checkpoint_v1"
DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf012_oracle_legal_bc_dataset_64"
)
DEFAULT_OUTPUT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / TAG
)
DATASET_REPORT_SHA256 = (
    "9bb31d59d126b00a037566a7a5745af827037740f91c022a070e4b953de5746f"
)
DATASET_METADATA_SHA256 = (
    "21158fbcdcf749c54b459991e379ee315edf71ab2b0c1efaa3f73e38a7f2fbb2"
)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42013
    hidden_size: int = 256
    initial_std: float = 0.15
    epochs: int = 8
    agent_batch_size: int = 8
    sequence_chunk: int = 64
    validation_agents: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    smoothness_weight: float = 1e-4
    action_weights: tuple[float, float, float, float] = (1.0, 1.0, 4.0, 1.0)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_agent_split(
    agents: int, validation_agents: int
) -> tuple[np.ndarray, np.ndarray]:
    if agents <= 1 or validation_agents <= 0 or validation_agents >= agents:
        raise ValueError("validation split must leave nonempty train and validation sets")
    boundary = agents - validation_agents
    return np.arange(boundary, dtype=np.int64), np.arange(
        boundary, agents, dtype=np.int64
    )


def reconstruct_legal_batch(
    mask: np.ndarray,
    tail: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Decode time-major legal storage to a batch-major actor tensor."""

    if mask.ndim != 3 or mask.shape[-1] != MASK_SIZE or mask.dtype != np.uint8:
        raise ValueError("mask batch must be uint8 [time,batch,4096]")
    if (
        tail.ndim != 3
        or tail.shape[:2] != mask.shape[:2]
        or tail.shape[-1] != LEGAL_OBS_SIZE - MASK_SIZE
    ):
        raise ValueError("legal tail does not align with the mask batch")
    decoded_mask = torch.from_numpy(np.asarray(mask).copy()).float().mul_(1.0 / 255.0)
    decoded_tail = torch.from_numpy(np.asarray(tail, dtype=np.float32).copy())
    legal = torch.cat((decoded_mask, decoded_tail), -1).transpose(0, 1).contiguous()
    return legal.to(device, non_blocking=False)


def weighted_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    action_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or prediction.shape[-1] != ACTION_SIZE:
        raise ValueError("prediction and target must align on four actions")
    if valid.shape != prediction.shape[:-1]:
        raise ValueError("valid mask does not align with action sequence")
    if action_weights.shape != (ACTION_SIZE,):
        raise ValueError("action weight vector must have four values")
    valid_float = valid.to(prediction.dtype).unsqueeze(-1)
    count = valid_float.sum().clamp_min(1.0)
    channel_mse = ((prediction - target).square() * valid_float).sum((0, 1)) / count
    loss = (channel_mse * action_weights).sum() / action_weights.sum()
    return loss, channel_mse


def temporal_smoothness(
    prediction: torch.Tensor,
    valid: torch.Tensor,
    *,
    previous_prediction: torch.Tensor | None,
    previous_valid: torch.Tensor | None,
) -> torch.Tensor:
    differences: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    if prediction.shape[1] > 1:
        differences.append(prediction[:, 1:] - prediction[:, :-1])
        masks.append(valid[:, 1:] & valid[:, :-1])
    if previous_prediction is not None:
        if previous_valid is None or previous_prediction.shape != prediction[:, 0].shape:
            raise ValueError("previous prediction boundary is incomplete")
        differences.append((prediction[:, 0] - previous_prediction).unsqueeze(1))
        masks.append((valid[:, 0] & previous_valid).unsqueeze(1))
    if not differences:
        return prediction.sum() * 0.0
    delta = torch.cat(differences, 1)
    mask = torch.cat(masks, 1).to(prediction.dtype).unsqueeze(-1)
    denominator = (mask.sum() * prediction.shape[-1]).clamp_min(1.0)
    return (delta.square() * mask).sum() / denominator


class OracleBCDataset:
    def __init__(
        self,
        root: Path,
        *,
        verify_hashes: bool = True,
        expected_report_sha256: str = DATASET_REPORT_SHA256,
        expected_metadata_sha256: str = DATASET_METADATA_SHA256,
    ) -> None:
        self.root = root
        if sha256_path(root / "report.json") != expected_report_sha256:
            raise RuntimeError("legal dataset report hash mismatch")
        if sha256_path(root / "metadata.json") != expected_metadata_sha256:
            raise RuntimeError("legal dataset metadata hash mismatch")
        self.report = json.loads((root / "report.json").read_text())
        self.metadata = json.loads((root / "metadata.json").read_text())
        if not self.report.get("admitted"):
            raise RuntimeError("SF012 dataset was not admitted")
        observation = self.metadata.get("observation", {})
        if (
            observation.get("stored_legal_width") != LEGAL_OBS_SIZE
            or observation.get("legal_tail_width") != LEGAL_OBS_SIZE - MASK_SIZE
            or observation.get("stored_privileged_values_per_record") != 0
        ):
            raise RuntimeError("SF012 actor storage boundary changed")
        if verify_hashes:
            for name, contract in self.metadata["files"].items():
                if sha256_path(root / name) != contract["sha256"]:
                    raise RuntimeError(f"SF012 data hash mismatch for {name}")
        self.mask = np.load(root / "mask.npy", mmap_mode="r")
        self.tail = np.load(root / "tail.npy", mmap_mode="r")
        self.action = np.load(root / "action.npy", mmap_mode="r")
        self.terminal = np.load(root / "terminal.npy", mmap_mode="r")
        self.valid = np.load(root / "valid.npy", mmap_mode="r")
        self.lengths = np.asarray(self.metadata["episode_lengths"], dtype=np.int64)
        self.time_steps, self.agents = self.valid.shape
        expected = {
            "mask": (self.time_steps, self.agents, MASK_SIZE),
            "tail": (
                self.time_steps,
                self.agents,
                LEGAL_OBS_SIZE - MASK_SIZE,
            ),
            "action": (self.time_steps, self.agents, ACTION_SIZE),
            "terminal": (self.time_steps, self.agents),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise RuntimeError(f"SF012 {name} shape changed")
        if self.lengths.shape != (self.agents,):
            raise RuntimeError("SF012 episode lengths do not align with agents")

    def chunk(
        self,
        agent_indices: np.ndarray,
        start: int,
        end: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.asarray(agent_indices, dtype=np.int64)
        if start < 0 or end <= start or end > self.time_steps:
            raise ValueError("invalid sequence chunk")
        mask = np.take(self.mask[start:end], indices, axis=1)
        tail = np.take(self.tail[start:end], indices, axis=1)
        legal = reconstruct_legal_batch(mask, tail, device=device)
        action_np = np.take(self.action[start:end], indices, axis=1)
        valid_np = np.take(self.valid[start:end], indices, axis=1)
        action = torch.from_numpy(np.asarray(action_np, dtype=np.float32).copy())
        action = action.transpose(0, 1).contiguous().to(device)
        valid = torch.from_numpy(np.asarray(valid_np, dtype=np.uint8).copy())
        valid = valid.transpose(0, 1).contiguous().bool().to(device)
        return legal, action, valid


class ActionAccumulator:
    def __init__(self) -> None:
        self.square = np.zeros(ACTION_SIZE, dtype=np.float64)
        self.absolute = np.zeros(ACTION_SIZE, dtype=np.float64)
        self.maximum = np.zeros(ACTION_SIZE, dtype=np.float64)
        self.count = 0

    def add(
        self, prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> None:
        selected = (prediction - target)[valid].detach().float().cpu().numpy()
        if selected.size == 0:
            return
        self.square += np.square(selected).sum(0)
        self.absolute += np.abs(selected).sum(0)
        self.maximum = np.maximum(self.maximum, np.abs(selected).max(0))
        self.count += int(selected.shape[0])

    def result(self) -> dict[str, Any]:
        if self.count <= 0:
            raise RuntimeError("no valid actions reached the metric accumulator")
        return {
            "count": self.count,
            "mse": (self.square / self.count).tolist(),
            "mae": (self.absolute / self.count).tolist(),
            "max_abs_error": self.maximum.tolist(),
            "mean_mse": float(self.square.sum() / (self.count * ACTION_SIZE)),
        }


def _agent_batches(order: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(order), size):
        yield order[start : start + size]


def evaluate(
    model: VQ2RecurrentActor,
    dataset: OracleBCDataset,
    agents: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    metrics = ActionAccumulator()
    weights = torch.tensor(config.action_weights, device=device)
    weighted_square = 0.0
    weighted_count = 0
    with torch.no_grad():
        for batch_agents in _agent_batches(agents, config.agent_batch_size):
            state = model.initial_state(len(batch_agents), device=device)
            maximum = int(dataset.lengths[batch_agents].max())
            for start in range(0, maximum, config.sequence_chunk):
                end = min(start + config.sequence_chunk, maximum)
                legal, target, valid = dataset.chunk(
                    batch_agents, start, end, device=device
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    actor_output, state = model.forward_sequence(legal, state)
                    loss, _ = weighted_action_mse(
                        actor_output.mean, target, valid, weights
                    )
                count = int(valid.sum().item())
                weighted_square += float(loss.item()) * count
                weighted_count += count
                metrics.add(actor_output.mean, target, valid)
    result = metrics.result()
    result["weighted_mse"] = weighted_square / max(weighted_count, 1)
    return result


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: TrainConfig = TrainConfig(),
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF013 preregisters CUDA training but CUDA is unavailable")
    device = torch.device(device_name)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    dataset = OracleBCDataset(DATASET, verify_hashes=True)
    train_agents, validation_agents = actor_agent_split(
        dataset.agents, config.validation_agents
    )
    model = VQ2RecurrentActor(
        hidden_size=config.hidden_size,
        initial_std=config.initial_std,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = torch.tensor(config.action_weights, device=device)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    optimizer_updates = 0
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = rng.permutation(train_agents)
        train_metrics = ActionAccumulator()
        train_loss_sum = 0.0
        train_count = 0
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
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer_updates += 1
                state = next_state.detach()
                previous_prediction = actor_output.mean[:, -1].detach()
                previous_valid = valid[:, -1].detach()
                count = int(valid.sum().item())
                train_loss_sum += float(loss.detach().item()) * count
                train_count += count
                train_metrics.add(actor_output.mean, target, valid)

        validation = evaluate(
            model, dataset, validation_agents, config, device
        )
        epoch_report = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "train": train_metrics.result(),
            "validation": validation,
            "optimizer_updates": optimizer_updates,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        if validation["weighted_mse"] < best_validation:
            best_validation = float(validation["weighted_mse"])
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training produced no selectable checkpoint")
    wall_time = time.perf_counter() - started
    output.mkdir(parents=True)
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "docs/vq2_sf014_recurrent_bc_preregistration_2026-07-28.md",
        DATASET / "report.json",
        DATASET / "metadata.json",
    ]
    source_sha256 = {
        str(path.relative_to(ROOT)): sha256_path(path) for path in source_paths
    }
    checkpoint = {
        "schema": SCHEMA,
        "tag": TAG,
        "model": {
            "class": "VQ2RecurrentActor",
            "legal_observation_size": LEGAL_OBS_SIZE,
            "action_size": ACTION_SIZE,
            "hidden_size": config.hidden_size,
            "initial_std": config.initial_std,
        },
        "model_state": best_state,
        "train_config": asdict(config),
        "train_agents": train_agents.tolist(),
        "validation_agents": validation_agents.tolist(),
        "dataset_report_sha256": DATASET_REPORT_SHA256,
        "dataset_metadata_sha256": DATASET_METADATA_SHA256,
        "best_epoch": best_epoch,
        "best_validation_weighted_mse": best_validation,
        "history": history,
        "optimizer_updates": optimizer_updates,
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
    report = {
        "schema": "vq2_recurrent_bc_training_report_v1",
        "tag": TAG,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_weighted_mse": best_validation,
        "history": history,
        "optimizer_updates": optimizer_updates,
        "wall_time_seconds": wall_time,
        "train_records": int(dataset.lengths[train_agents].sum()),
        "validation_records": int(dataset.lengths[validation_agents].sum()),
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
