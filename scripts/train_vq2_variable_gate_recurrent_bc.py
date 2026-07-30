#!/usr/bin/env python3
"""Train one 4,119-input recurrent actor on the admitted VG003 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
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

from pufferlib.vq2_informed import MASK_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import (
    PHASE_LEGAL_OBS_SIZE,
    VQ2PhaseRecurrentActor,
)
from scripts.eval_vq2_variable_gate_oracle import write_json_atomic
from scripts.train_vq2_recurrent_bc import (
    ActionAccumulator,
    temporal_smoothness,
    weighted_action_mse,
)


TAG = "vq2_vg005_variable_gate_recurrent_bc_001"
SCHEMA = "vq2_variable_gate_recurrent_bc_checkpoint_v1"
DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg003_variable_gate_legal_bc_dataset_256"
)
DATASET_REPORT_SHA256 = (
    "b8069d05cec63038659893a4020fa61ca1276ae035bc1a2ea5754f75e2298e85"
)
DATASET_METADATA_SHA256 = (
    "7a5d6359641c3e01f838e6fcdb57d32ddc18cc71ddbf1f8b3d0025309cabfe64"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_vg005_variable_gate_recurrent_bc_preregistration_2026-07-29.md"
)
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)
PHASE_TAIL_INDEX = PHASE_LEGAL_OBS_SIZE - MASK_SIZE - 1


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 429031
    hidden_size: int = 256
    initial_std: float = 0.15
    epochs: int = 12
    agent_batch_size: int = 8
    sequence_chunk: int = 256
    validation_agents: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    smoothness_weight: float = 1e-4
    transition_window_exposure: int = 3
    action_weights: tuple[float, float, float, float] = (1.0, 1.0, 4.0, 1.0)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_label(path: Path) -> str:
    """Use repository-relative evidence labels, retaining explicit externals."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def runtime_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "numpy": np.__version__,
    }


def reconstruct_phase_batch(
    mask: np.ndarray,
    tail: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Decode time-major VG003 storage to batch-major actor input."""

    if mask.ndim != 3 or mask.shape[-1] != MASK_SIZE or mask.dtype != np.uint8:
        raise ValueError("mask batch must be uint8 [time,batch,4096]")
    if (
        tail.ndim != 3
        or tail.shape[:2] != mask.shape[:2]
        or tail.shape[-1] != PHASE_LEGAL_OBS_SIZE - MASK_SIZE
    ):
        raise ValueError("phase tail does not align with the mask batch")
    mask_array = np.ascontiguousarray(mask)
    tail_array = np.ascontiguousarray(tail, dtype=np.float32)
    if not mask_array.flags.writeable:
        mask_array = mask_array.copy()
    if not tail_array.flags.writeable:
        tail_array = tail_array.copy()

    # Keep the 4,096-byte visual mask compact across the host/device boundary.
    # Expanding it to float32 on the CPU made the loader dominate five-source
    # training while the GPU waited for synchronous copies.
    decoded_mask = torch.from_numpy(mask_array).to(device, non_blocking=False)
    decoded_mask = decoded_mask.float().mul_(1.0 / 255.0)
    decoded_tail = torch.from_numpy(tail_array).to(device, non_blocking=False)
    observation = torch.cat((decoded_mask, decoded_tail), -1)
    return observation.transpose(0, 1).contiguous()


def phase_increment_rows(
    phase: np.ndarray,
    valid: np.ndarray,
    previous_phase: np.ndarray,
) -> np.ndarray:
    """Mark each causal held-status increment, including a chunk boundary."""

    values = np.asarray(phase, dtype=np.float32)
    valid_values = np.asarray(valid, dtype=np.uint8)
    previous = np.asarray(previous_phase, dtype=np.float32)
    if values.ndim != 2 or valid_values.shape != values.shape:
        raise ValueError("phase and valid arrays must align [time,batch]")
    if previous.shape != (values.shape[1],):
        raise ValueError("previous phase must align with the agent batch")
    prior = np.concatenate((previous[None], values[:-1]), axis=0)
    increments = (values > prior + 1e-7) & (valid_values != 0)
    if np.any((values + 1e-7 < prior) & (valid_values != 0)):
        raise RuntimeError("public phase decreased inside the BC dataset")
    return increments


def actor_agent_split(
    agents: int,
    validation_agents: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve the final exact-uniform block and train on every prior agent."""

    if agents <= 1 or validation_agents <= 0 or validation_agents >= agents:
        raise ValueError("validation split must leave train and validation agents")
    boundary = agents - validation_agents
    return np.arange(boundary, dtype=np.int64), np.arange(
        boundary, agents, dtype=np.int64
    )


def audit_public_phase_layout(
    tail: np.ndarray,
    valid: np.ndarray,
    lengths: np.ndarray,
) -> dict[str, int | float]:
    """Audit every valid phase prefix while explicitly excluding padding."""

    if tail.ndim != 3 or tail.shape[-1] <= PHASE_TAIL_INDEX:
        raise ValueError("phase audit tail has the wrong ABI")
    if valid.shape != tail.shape[:2] or lengths.shape != (tail.shape[1],):
        raise ValueError("phase audit layout does not align")
    increments = 0
    encoding_error = 0.0
    invalid_padding_rows = 0
    for agent, raw_length in enumerate(lengths):
        length = int(raw_length)
        if length <= 0 or length > tail.shape[0]:
            raise RuntimeError("phase audit episode length escaped storage")
        agent_valid = np.asarray(valid[:, agent], dtype=np.uint8)
        if not np.all(agent_valid[:length] == 1) or np.any(
            agent_valid[length:] != 0
        ):
            raise RuntimeError("phase audit found a non-contiguous valid prefix")
        phase = np.asarray(
            tail[:length, agent, PHASE_TAIL_INDEX], dtype=np.float32
        )
        if not np.isfinite(phase).all():
            raise RuntimeError("phase audit found a non-finite value")
        scaled = phase.astype(np.float64) * 16.0
        encoding_error = max(
            encoding_error,
            float(np.max(np.abs(scaled - np.rint(scaled)), initial=0.0)),
        )
        delta = np.diff(np.concatenate((np.zeros(1, dtype=np.float32), phase)))
        if np.any(delta < -1e-7):
            raise RuntimeError("phase audit found a valid-row decrease")
        if np.any(delta > 1.0 / 16.0 + 1e-7):
            raise RuntimeError("phase audit found a skipped public index")
        increments += int((delta > 1e-7).sum())
        invalid_padding_rows += int(tail.shape[0] - length)
    if encoding_error > 1e-6:
        raise RuntimeError("phase audit found a non-/16 encoding")
    return {
        "increments": increments,
        "encoding_max_error": encoding_error,
        "invalid_padding_rows_excluded": invalid_padding_rows,
    }


def _agent_batches(order: np.ndarray, size: int) -> Iterable[np.ndarray]:
    if size <= 0:
        raise ValueError("agent batch size must be positive")
    for start in range(0, len(order), size):
        yield order[start : start + size]


class VariableGateBCDataset:
    def __init__(
        self,
        root: Path = DATASET,
        *,
        report_path: Path | None = None,
        verify_hashes: bool = True,
        expected_report_sha256: str = DATASET_REPORT_SHA256,
        expected_metadata_sha256: str = DATASET_METADATA_SHA256,
    ) -> None:
        self.root = Path(root)
        self.report_path = (
            self.root / "report.json"
            if report_path is None
            else Path(report_path)
        )
        if sha256_path(self.report_path) != expected_report_sha256:
            raise RuntimeError("VG003 dataset report hash mismatch")
        if sha256_path(self.root / "metadata.json") != expected_metadata_sha256:
            raise RuntimeError("VG003 dataset metadata hash mismatch")
        self.report = json.loads(self.report_path.read_text())
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        observation = self.metadata.get("observation", {})
        if not self.report.get("admitted") or (
            observation.get("stored_legal_width") != PHASE_LEGAL_OBS_SIZE
            or observation.get("mask_width") != MASK_SIZE
            or observation.get("legal_sensor_history_width") != 22
            or observation.get("public_phase_width") != 1
            or observation.get("public_phase_tail_index") != 22
            or observation.get("public_phase_encoding")
            != "clamp(active_gate_index,0,16)/16"
            or observation.get("public_phase_rate_hz") != 4
            or observation.get("stored_training_only_privileged_values_per_record")
            != 0
            or observation.get("stored_total_gate_count_values_per_record") != 0
        ):
            raise RuntimeError("VG003 actor storage boundary changed")
        if verify_hashes:
            for name, contract in self.metadata["files"].items():
                if sha256_path(self.root / name) != contract["sha256"]:
                    raise RuntimeError(f"VG003 data hash mismatch for {name}")
        self.mask = np.load(self.root / "mask.npy", mmap_mode="r")
        self.tail = np.load(self.root / "tail.npy", mmap_mode="r")
        self.action = np.load(self.root / "action.npy", mmap_mode="r")
        self.terminal = np.load(self.root / "terminal.npy", mmap_mode="r")
        self.valid = np.load(self.root / "valid.npy", mmap_mode="r")
        self.lengths = np.asarray(
            self.metadata["episode_lengths"], dtype=np.int64
        )
        self.time_steps, self.agents = self.valid.shape
        expected = {
            "mask": (self.time_steps, self.agents, MASK_SIZE),
            "tail": (
                self.time_steps,
                self.agents,
                PHASE_LEGAL_OBS_SIZE - MASK_SIZE,
            ),
            "action": (self.time_steps, self.agents, ACTION_SIZE),
            "terminal": (self.time_steps, self.agents),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise RuntimeError(f"VG003 {name} shape changed")
        if self.lengths.shape != (self.agents,):
            raise RuntimeError("VG003 episode lengths do not align with agents")
        self.phase_audit = audit_public_phase_layout(
            self.tail, self.valid, self.lengths
        )
        expected_increments = self.report.get("dataset", {}).get(
            "phase_increments"
        )
        if (
            expected_increments is not None
            and self.phase_audit["increments"] != expected_increments
        ):
            raise RuntimeError("VG003 phase audit differs from collection report")

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
            raise ValueError("invalid VG003 sequence chunk")
        mask = np.take(self.mask[start:end], indices, axis=1)
        tail = np.take(self.tail[start:end], indices, axis=1)
        valid_np = np.take(self.valid[start:end], indices, axis=1)
        previous_phase = (
            np.zeros(len(indices), dtype=np.float32)
            if start == 0
            else np.take(self.tail[start - 1, :, PHASE_TAIL_INDEX], indices)
        )
        transition_np = phase_increment_rows(
            tail[:, :, PHASE_TAIL_INDEX], valid_np, previous_phase
        )
        observation = reconstruct_phase_batch(mask, tail, device=device)
        action_np = np.take(self.action[start:end], indices, axis=1)
        action = torch.from_numpy(np.ascontiguousarray(action_np, dtype=np.float32))
        action = action.transpose(0, 1).contiguous().to(device)
        valid = torch.from_numpy(np.ascontiguousarray(valid_np, dtype=np.uint8))
        valid = valid.transpose(0, 1).contiguous().bool().to(device)
        transition = torch.from_numpy(np.ascontiguousarray(transition_np))
        transition = transition.transpose(0, 1).contiguous().bool().to(device)
        return observation, action, valid, transition


def evaluate(
    model: VQ2PhaseRecurrentActor,
    dataset: VariableGateBCDataset,
    agents: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    overall = ActionAccumulator()
    transitions = ActionAccumulator()
    weights = torch.tensor(config.action_weights, device=device)
    weighted_sum = 0.0
    weighted_count = 0
    transition_rows = 0
    with torch.no_grad():
        for batch_agents in _agent_batches(agents, config.agent_batch_size):
            state = model.initial_state(len(batch_agents), device=device)
            maximum = int(dataset.lengths[batch_agents].max())
            for start in range(0, maximum, config.sequence_chunk):
                end = min(start + config.sequence_chunk, maximum)
                observation, target, valid, transition = dataset.chunk(
                    batch_agents, start, end, device=device
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    output, state = model.forward_sequence(observation, state)
                    loss, _ = weighted_action_mse(
                        output.mean, target, valid, weights
                    )
                count = int(valid.sum().item())
                weighted_sum += float(loss.item()) * count
                weighted_count += count
                overall.add(output.mean, target, valid)
                transitions.add(output.mean, target, transition)
                transition_rows += int(transition.sum().item())
    result = overall.result()
    result["weighted_mse"] = weighted_sum / max(weighted_count, 1)
    result["transition_rows"] = transition_rows
    result["transition"] = transitions.result()
    return result


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _state_payload(
    *,
    identity: dict[str, Any],
    status: str,
    completed_epoch: int,
    model: VQ2PhaseRecurrentActor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    rng: np.random.Generator,
    history: list[dict[str, Any]],
    best_epoch: int,
    best_validation: float,
    best_state: dict[str, torch.Tensor] | None,
    optimizer_updates: int,
) -> dict[str, Any]:
    return {
        **identity,
        "status": status,
        "completed_epoch": completed_epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "numpy_rng_state": rng.bit_generator.state,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_weighted_mse": best_validation,
        "best_state": best_state,
        "optimizer_updates": optimizer_updates,
    }


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: TrainConfig = TrainConfig(),
    resume: bool = False,
) -> dict[str, Any]:
    if config.sequence_chunk < 256:
        raise RuntimeError("VG005 requires BPTT windows of at least 256 steps")
    if config.transition_window_exposure < 3:
        raise RuntimeError("VG005 requires at least 3x transition-window exposure")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG005 preregistration is missing")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG005 preregisters CUDA training")
    report_path = output / "report.json"
    state_path = output / "training_state.pt"
    existing_report: dict[str, Any] | None = None
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        existing_report = json.loads(report_path.read_text())
        if existing_report.get("completed") is not True:
            raise RuntimeError("existing VG005 report is not completed")
    if output.exists() and not state_path.is_file():
        raise RuntimeError("VG005 output exists without resumable state")

    device = torch.device(device_name)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    source_paths = [
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        ROOT / "scripts/train_vq2_recurrent_bc.py",
        ROOT / "scripts/run_vq2_vg005_vast.sh",
        DATASET / "report.json",
        DATASET / "metadata.json",
    ]
    source_sha256 = {
        source_label(path): sha256_path(path) for path in source_paths
    }
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dataset = VariableGateBCDataset(
        DATASET,
        verify_hashes=True,
        expected_report_sha256=DATASET_REPORT_SHA256,
        expected_metadata_sha256=DATASET_METADATA_SHA256,
    )
    identity = {
        "schema": "vq2_variable_gate_recurrent_bc_state_v1",
        "tag": TAG,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": runtime_manifest(),
        "train_config": asdict(config),
        "dataset_report_sha256": DATASET_REPORT_SHA256,
        "dataset_metadata_sha256": DATASET_METADATA_SHA256,
        "dataset_phase_audit": dataset.phase_audit,
        "safety": {
            "actor_input_privileged_values": 0,
            "teacher_blend": 0.0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }

    if existing_report is not None:
        expected_report_values = {
            "source_commit": source_commit,
            "source_sha256": source_sha256,
            "runtime": identity["runtime"],
            "train_config": asdict(config),
            "dataset_report_sha256": DATASET_REPORT_SHA256,
            "dataset_metadata_sha256": DATASET_METADATA_SHA256,
            "dataset_phase_audit": dataset.phase_audit,
        }
        for key, expected in expected_report_values.items():
            if existing_report.get(key) != expected:
                raise RuntimeError(f"VG005 completed report mismatch for {key}")
        return existing_report

    train_agents, validation_agents = actor_agent_split(
        dataset.agents, config.validation_agents
    )
    model = VQ2PhaseRecurrentActor(
        hidden_size=config.hidden_size,
        initial_std=config.initial_std,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    optimizer_updates = 0
    completed_epoch = 0

    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"VG005 state already exists at {state_path}")
        saved = torch.load(state_path, map_location=device, weights_only=False)
        for key, expected in identity.items():
            if saved.get(key) != expected:
                raise RuntimeError(f"VG005 resume mismatch for {key}")
        if saved.get("status") != "training":
            raise RuntimeError("VG005 can resume only active training state")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        random.setstate(saved["python_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        history = saved["history"]
        best_validation = float(saved["best_validation_weighted_mse"])
        best_epoch = int(saved["best_epoch"])
        best_state = saved["best_state"]
        optimizer_updates = int(saved["optimizer_updates"])
        completed_epoch = int(saved["completed_epoch"])
    else:
        if resume:
            raise RuntimeError("VG005 --resume requested without run state")
        output.mkdir(parents=True)
        atomic_torch_save(
            state_path,
            _state_payload(
                identity=identity,
                status="training",
                completed_epoch=0,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                rng=rng,
                history=history,
                best_epoch=best_epoch,
                best_validation=best_validation,
                best_state=best_state,
                optimizer_updates=optimizer_updates,
            ),
        )

    weights = torch.tensor(config.action_weights, device=device)
    started = time.perf_counter()
    for epoch in range(completed_epoch + 1, config.epochs + 1):
        model.train()
        metrics = ActionAccumulator()
        loss_sum = 0.0
        exposed_records = 0
        base_agent_windows = 0
        transition_agent_windows = 0
        transition_agent_window_exposures = 0
        transition_batch_windows = 0
        for batch_agents in _agent_batches(
            rng.permutation(train_agents), config.agent_batch_size
        ):
            state = model.initial_state(len(batch_agents), device=device)
            previous_prediction: torch.Tensor | None = None
            previous_valid: torch.Tensor | None = None
            maximum = int(dataset.lengths[batch_agents].max())
            for start in range(0, maximum, config.sequence_chunk):
                end = min(start + config.sequence_chunk, maximum)
                observation, target, valid, transition = dataset.chunk(
                    batch_agents, start, end, device=device
                )
                active_windows = valid.any(dim=1)
                transition_windows = transition.any(dim=1)
                base_agent_windows += int(active_windows.sum().item())
                transition_count = int(transition_windows.sum().item())
                transition_agent_windows += transition_count
                repetitions = (
                    config.transition_window_exposure
                    if transition_count > 0
                    else 1
                )
                if transition_count > 0:
                    transition_batch_windows += 1
                transition_agent_window_exposures += transition_count * repetitions
                start_state = state.detach()
                final_output = None
                final_state = None
                for _ in range(repetitions):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda",
                    ):
                        output_value, next_state = model.forward_sequence(
                            observation, start_state
                        )
                        action_loss, _ = weighted_action_mse(
                            output_value.mean, target, valid, weights
                        )
                        smoothness = temporal_smoothness(
                            output_value.mean,
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
                    count = int(valid.sum().item())
                    loss_sum += float(loss.detach().item()) * count
                    exposed_records += count
                    final_output = output_value
                    final_state = next_state
                if final_output is None or final_state is None:
                    raise RuntimeError("VG005 emitted no training update")
                metrics.add(final_output.mean, target, valid)
                state = final_state.detach()
                previous_prediction = final_output.mean[:, -1].detach()
                previous_valid = valid[:, -1].detach()

        validation = evaluate(
            model, dataset, validation_agents, config, device
        )
        minimum_transition_exposure = (
            transition_agent_window_exposures / transition_agent_windows
            if transition_agent_windows
            else 0.0
        )
        epoch_report = {
            "epoch": epoch,
            "train_loss": loss_sum / max(exposed_records, 1),
            "train": metrics.result(),
            "validation": validation,
            "optimizer_updates": optimizer_updates,
            "base_agent_windows": base_agent_windows,
            "transition_agent_windows": transition_agent_windows,
            "transition_batch_windows": transition_batch_windows,
            "transition_agent_window_exposures": transition_agent_window_exposures,
            "minimum_transition_window_exposure": minimum_transition_exposure,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        score = float(validation["weighted_mse"])
        if np.isfinite(score) and score < best_validation:
            best_validation = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        completed_epoch = epoch
        atomic_torch_save(
            state_path,
            _state_payload(
                identity=identity,
                status="training",
                completed_epoch=completed_epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                rng=rng,
                history=history,
                best_epoch=best_epoch,
                best_validation=best_validation,
                best_state=best_state,
                optimizer_updates=optimizer_updates,
            ),
        )

    if best_state is None:
        raise RuntimeError("VG005 produced no finite checkpoint")
    minimum_exposure = min(
        float(epoch["minimum_transition_window_exposure"]) for epoch in history
    )
    if minimum_exposure + 1e-12 < config.transition_window_exposure:
        raise RuntimeError("VG005 did not meet transition-window exposure")

    checkpoint = {
        "schema": SCHEMA,
        "tag": TAG,
        "model": {
            "class": "VQ2PhaseRecurrentActor",
            "legal_observation_size": PHASE_LEGAL_OBS_SIZE,
            "legacy_legal_observation_size": PHASE_LEGAL_OBS_SIZE - 1,
            "public_status_values": 1,
            "public_phase_encoding": "clamp(active_gate_index,0,16)/16",
            "public_phase_rate_hz": 4,
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
        "dataset_phase_audit": dataset.phase_audit,
        "best_epoch": best_epoch,
        "best_validation_weighted_mse": best_validation,
        "history": history,
        "optimizer_updates": optimizer_updates,
        "minimum_transition_window_exposure": minimum_exposure,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": identity["runtime"],
        "safety": identity["safety"],
    }
    checkpoint_path = output / "policy_best.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": "vq2_variable_gate_recurrent_bc_training_report_v1",
        "tag": TAG,
        "completed": True,
        "checkpoint": source_label(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_weighted_mse": best_validation,
        "minimum_transition_window_exposure": minimum_exposure,
        "optimizer_updates": optimizer_updates,
        "history": history,
        "train_config": asdict(config),
        "dataset_report_sha256": DATASET_REPORT_SHA256,
        "dataset_metadata_sha256": DATASET_METADATA_SHA256,
        "dataset_phase_audit": dataset.phase_audit,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": identity["runtime"],
        "wall_time_seconds": time.perf_counter() - started,
        "safety": identity["safety"],
    }
    write_json_atomic(report_path, report)
    final_state = _state_payload(
        identity=identity,
        status="completed",
        completed_epoch=completed_epoch,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        rng=rng,
        history=history,
        best_epoch=best_epoch,
        best_validation=best_validation,
        best_state=best_state,
        optimizer_updates=optimizer_updates,
    )
    final_state["checkpoint_sha256"] = report["checkpoint_sha256"]
    final_state["report_sha256"] = sha256_path(report_path)
    atomic_torch_save(state_path, final_state)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train(
        output=args.output.resolve(),
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
