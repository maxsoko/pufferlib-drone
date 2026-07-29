#!/usr/bin/env python3
"""Fit only the public-phase input path while preserving SF033 at phase zero."""

from __future__ import annotations

import argparse
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
from pufferlib.vq2_recurrent_phase import (
    PHASE_LEGAL_OBS_SIZE,
    VQ2PhaseRecurrentActor,
)
from scripts.collect_vq2_oracle_bc_dataset import sha256_path
from scripts.train_vq2_recurrent_bc import (
    ActionAccumulator,
    actor_agent_split,
    temporal_smoothness,
    weighted_action_mse,
)


TAG = "vq2_sf040_public_phase_adapter_001"
SCHEMA = "vq2_public_phase_adapter_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
PREREGISTRATION = (
    ROOT / "docs/vq2_sf040_public_phase_adapter_preregistration_2026-07-28.md"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf033_recurrent_roll_priority_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "91088ca96f432f57f1b85bbc521ddab29ba422fadf2b79b8d642ee426f09bea7"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "f54bf8512490a4629c08bf35ceae7c7fe07514e3424f89433f680536d96a0279"
)
DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_sf039_public_phase_dagger_512"
)
DATASET_REPORT_SHA256 = (
    "812c1a637db10d199df9a27ee63a76320084b6287c6bfc0e3d388fd18eb8839a"
)
DATASET_METADATA_SHA256 = (
    "100ceaa25a975559d73dedf6642f93081ce09f7134be76b00ac533046f0f02f0"
)
CAUSAL_HORIZON_STEPS = 768
GATE2_PHASE = np.float32(1.0 / 6.0)


@dataclass(frozen=True)
class PhaseAdapterConfig:
    seed: int = 42040
    hidden_size: int = 256
    initial_std: float = 0.15
    epochs: int = 12
    agent_batch_size: int = 8
    sequence_chunk: int = 64
    validation_agents: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    smoothness_weight: float = 1e-4
    action_weights: tuple[float, float, float, float] = (4.0, 4.0, 4.0, 1.0)


def _agent_batches(order: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(order), size):
        yield order[start : start + size]


def reconstruct_phase_batch(
    mask: np.ndarray,
    tail: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    if mask.ndim != 3 or mask.shape[-1] != MASK_SIZE or mask.dtype != np.uint8:
        raise ValueError("mask batch must be uint8 [time,batch,4096]")
    if (
        tail.ndim != 3
        or tail.shape[:2] != mask.shape[:2]
        or tail.shape[-1] != PHASE_LEGAL_OBS_SIZE - MASK_SIZE
    ):
        raise ValueError("public-phase tail does not align with the mask batch")
    decoded_mask = torch.from_numpy(np.asarray(mask).copy()).float().mul_(1.0 / 255.0)
    decoded_tail = torch.from_numpy(np.asarray(tail, dtype=np.float32).copy())
    observation = torch.cat((decoded_mask, decoded_tail), -1)
    observation = observation.transpose(0, 1).contiguous()
    return observation.to(device, non_blocking=False)


class PublicPhaseDataset:
    def __init__(
        self,
        root: Path = DATASET,
        *,
        verify_hashes: bool = True,
        expected_report_sha256: str = DATASET_REPORT_SHA256,
        expected_metadata_sha256: str = DATASET_METADATA_SHA256,
        horizon: int = CAUSAL_HORIZON_STEPS,
    ) -> None:
        self.root = Path(root)
        if sha256_path(self.root / "report.json") != expected_report_sha256:
            raise RuntimeError("public-phase dataset report hash mismatch")
        if sha256_path(self.root / "metadata.json") != expected_metadata_sha256:
            raise RuntimeError("public-phase dataset metadata hash mismatch")
        self.report = json.loads((self.root / "report.json").read_text())
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        observation = self.metadata.get("observation", {})
        if not self.report.get("admitted") or (
            observation.get("stored_legal_width") != PHASE_LEGAL_OBS_SIZE
            or observation.get("legacy_legal_width") != LEGAL_OBS_SIZE
            or observation.get("legal_tail_width")
            != PHASE_LEGAL_OBS_SIZE - MASK_SIZE
            or observation.get("public_status_values_per_record") != 1
            or observation.get("stored_training_only_privileged_values_per_record")
            != 0
        ):
            raise RuntimeError("public-phase actor storage boundary changed")
        if verify_hashes:
            for name, contract in self.metadata["files"].items():
                if sha256_path(self.root / name) != contract["sha256"]:
                    raise RuntimeError(f"public-phase data hash mismatch for {name}")
        self.mask = np.load(self.root / "mask.npy", mmap_mode="r")
        self.tail = np.load(self.root / "tail.npy", mmap_mode="r")
        self.action = np.load(self.root / "action.npy", mmap_mode="r")
        self.valid = np.load(self.root / "valid.npy", mmap_mode="r")
        full_lengths = np.asarray(
            self.metadata["episode_lengths"], dtype=np.int64
        )
        full_steps, self.agents = self.valid.shape
        if horizon <= 0 or horizon > full_steps:
            raise ValueError("public-phase causal horizon is outside the dataset")
        self.time_steps = horizon
        self.lengths = np.minimum(full_lengths, horizon)
        expected = {
            "mask": (full_steps, self.agents, MASK_SIZE),
            "tail": (
                full_steps,
                self.agents,
                PHASE_LEGAL_OBS_SIZE - MASK_SIZE,
            ),
            "action": (full_steps, self.agents, ACTION_SIZE),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise RuntimeError(f"public-phase {name} shape changed")
        if full_lengths.shape != (self.agents,):
            raise RuntimeError("public-phase lengths do not align with agents")

    def chunk(
        self,
        agent_indices: np.ndarray,
        start: int,
        end: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.asarray(agent_indices, dtype=np.int64)
        if start < 0 or end <= start or end > self.time_steps:
            raise ValueError("invalid public-phase sequence chunk")
        mask = np.take(self.mask[start:end], indices, axis=1)
        tail = np.take(self.tail[start:end], indices, axis=1)
        observation = reconstruct_phase_batch(mask, tail, device=device)
        action_np = np.take(self.action[start:end], indices, axis=1)
        valid_np = np.take(self.valid[start:end], indices, axis=1)
        action = torch.from_numpy(np.asarray(action_np, dtype=np.float32).copy())
        action = action.transpose(0, 1).contiguous().to(device)
        valid = torch.from_numpy(np.asarray(valid_np, dtype=np.uint8).copy())
        valid = valid.transpose(0, 1).contiguous().bool().to(device)
        phase = observation[..., -1]
        gate2 = valid & torch.isclose(
            phase, torch.tensor(float(GATE2_PHASE), device=device), atol=1e-7, rtol=0.0
        )
        return observation, action, valid, gate2


def evaluate_gate2(
    model: VQ2PhaseRecurrentActor,
    dataset: PublicPhaseDataset,
    agents: np.ndarray,
    config: PhaseAdapterConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    metrics = ActionAccumulator()
    weights = torch.tensor(config.action_weights, device=device)
    weighted_sum = 0.0
    count_sum = 0
    with torch.no_grad():
        for batch_agents in _agent_batches(agents, config.agent_batch_size):
            state = model.initial_state(len(batch_agents), device=device)
            maximum = int(dataset.lengths[batch_agents].max())
            for start in range(0, maximum, config.sequence_chunk):
                end = min(start + config.sequence_chunk, maximum)
                observation, target, _, gate2 = dataset.chunk(
                    batch_agents, start, end, device=device
                )
                output, state = model.forward_sequence(observation, state)
                if not bool(gate2.any()):
                    continue
                loss, _ = weighted_action_mse(output.mean, target, gate2, weights)
                count = int(gate2.sum().item())
                weighted_sum += float(loss.item()) * count
                count_sum += count
                metrics.add(output.mean, target, gate2)
    result = metrics.result()
    result["weighted_mse"] = weighted_sum / max(count_sum, 1)
    return result


def phase_zero_is_exact(
    model: VQ2PhaseRecurrentActor,
    parent: VQ2RecurrentActor,
    dataset: PublicPhaseDataset,
    agents: np.ndarray,
    device: torch.device,
) -> bool:
    selected = agents[: min(len(agents), 8)]
    observation, _, valid, _ = dataset.chunk(selected, 0, 64, device=device)
    observation = observation.clone()
    observation[..., -1] = 0.0
    parent.eval()
    model.eval()
    with torch.no_grad():
        parent_output, parent_state = parent.forward_sequence(
            observation[..., :LEGAL_OBS_SIZE]
        )
        phase_output, phase_state = model.forward_sequence(observation)
    return bool(
        torch.equal(parent_output.mean[valid], phase_output.mean[valid])
        and torch.equal(parent_state, phase_state)
    )


def numerical_admission(metrics: dict[str, Any], phase_zero_exact: bool) -> bool:
    values = [float(metrics["weighted_mse"]), *map(float, metrics["mse"])]
    return bool(
        phase_zero_exact
        and np.isfinite(values).all()
        and values[0] <= 0.02
        and all(value <= 0.05 for value in values[1:])
    )


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: PhaseAdapterConfig = PhaseAdapterConfig(),
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SF040 preregisters CUDA training")
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
    model = VQ2PhaseRecurrentActor(
        hidden_size=config.hidden_size, initial_std=config.initial_std
    ).to(device)
    model.load_phase_zero_base_state(payload["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.phase_embedding.weight.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in trainable) != config.hidden_size:
        raise RuntimeError("SF040 must train exactly the 256-value phase embedding")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = torch.tensor(config.action_weights, device=device)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
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
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("SF040 produced no finite phase-adapter checkpoint")

    output.mkdir(parents=True)
    source_paths = [
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_informed.py",
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
            "class": "VQ2PhaseRecurrentActor",
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
        "causal_horizon_steps": CAUSAL_HORIZON_STEPS,
        "trainable_parameter_names": ["phase_embedding.weight"],
        "trainable_parameters": config.hidden_size,
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
    checkpoint_path = output / "policy_best.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": "vq2_public_phase_adapter_training_report_v1",
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
        "trainable_parameters": config.hidden_size,
        "history": history,
        "optimizer_updates": updates,
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
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
