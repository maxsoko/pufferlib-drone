#!/usr/bin/env python3
"""Source-balance VG003 clean anchors and VG009 visited-state labels."""

from __future__ import annotations

import argparse
import json
import os
import platform
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
from scripts.train_vq2_variable_gate_recurrent_bc import (
    VariableGateBCDataset,
    _agent_batches,
    atomic_torch_save,
    evaluate,
    sha256_path,
    source_label,
)


TAG = "vq2_vg010_variable_gate_source_balanced_refit_001"
SCHEMA = "vq2_variable_gate_recurrent_bc_checkpoint_v1"
SEED = 429050
CLEAN_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg003_variable_gate_legal_bc_dataset_256"
)
CLEAN_REPORT_SHA256 = (
    "b8069d05cec63038659893a4020fa61ca1276ae035bc1a2ea5754f75e2298e85"
)
CLEAN_METADATA_SHA256 = (
    "7a5d6359641c3e01f838e6fcdb57d32ddc18cc71ddbf1f8b3d0025309cabfe64"
)
DAGGER_DATASET = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg009_variable_gate_dagger_round1_corrected_512"
)
DAGGER_REPORT_SHA256 = (
    "72c1e41d128f16ff082362e0dac2fb0cea36eef1a7d65e8e96f8ec31c3f50637"
)
DAGGER_METADATA_SHA256 = (
    "da20bd4ef93b7371712880787226a7f41b23785841d7307d501e0b7a8f829468"
)
DAGGER_ADMISSION = (
    ROOT / "docs/vq2_vg009_variable_gate_dagger_round1_admission_2026-07-30.json"
)
DAGGER_ADMISSION_SHA256 = (
    "71c4898f7a7a306b0f6def06bda63d5192de456c8a849e1f08f83ba908cd3091"
)
PARENT_CHECKPOINT = (
    ROOT
    / "logs/drone_race_full_policy_six_gate_bootstrap"
    / "vq2_vg005_variable_gate_recurrent_bc_001/policy_best.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "f686a35dc923137bc40fd6feca459e52e5585d5b6fa047b7f77a171e2832a6c3"
)
PARENT_REPORT = PARENT_CHECKPOINT.parent / "report.json"
PARENT_REPORT_SHA256 = (
    "bdcd2b38ea3537f2c250281c63f2e6aeca0ad87c97521339f098eec7ff474e5b"
)
GOAL_PROMPT = ROOT / "docs/vq2_variable_gate_solve_first_goal_prompt_2026-07-28.md"
GOAL_PROMPT_SHA256 = (
    "052ba7cb7a6db0b0274731f555a726994f83d424cb50e70f055e804926333e89"
)
PREREGISTRATION = (
    ROOT / "docs/vq2_vg010_variable_gate_dagger_refit_preregistration_2026-07-30.md"
)
DEFAULT_OUTPUT = (
    ROOT / "logs/drone_race_full_policy_six_gate_bootstrap" / TAG
)


@dataclass(frozen=True)
class RefitConfig:
    seed: int = SEED
    hidden_size: int = 256
    initial_std: float = 0.15
    epochs: int = 12
    source_agent_batch_size: int = 4
    agent_batch_size: int = 8
    sequence_chunk: int = 256
    clean_validation_agents: int = 32
    dagger_validation_agents: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    smoothness_weight: float = 1e-4
    transition_window_exposure: int = 3
    clean_objective_weight: float = 0.5
    dagger_objective_weight: float = 0.5
    action_weights: tuple[float, float, float, float] = (1.0, 1.0, 4.0, 1.0)
    maximum_clean_validation_weighted_mse: float = 0.02


@dataclass
class StreamItem:
    observation: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor
    transition: torch.Tensor
    start_state: torch.Tensor
    previous_prediction: torch.Tensor | None
    previous_valid: torch.Tensor | None


class RecurrentChunkStream:
    """Emit contiguous source chunks and preserve each batch's recurrent state."""

    def __init__(
        self,
        dataset: VariableGateBCDataset,
        agents: np.ndarray,
        *,
        batch_size: int,
        sequence_chunk: int,
        rng: np.random.Generator,
        device: torch.device,
        cyclic: bool,
    ) -> None:
        self.dataset = dataset
        self.agents = np.asarray(agents, dtype=np.int64)
        self.batch_size = batch_size
        self.sequence_chunk = sequence_chunk
        self.rng = rng
        self.device = device
        self.cyclic = cyclic
        self.cycles_started = 0
        self.source_chunks = 0
        self.source_records = 0
        self._batches: list[np.ndarray] = []
        self._batch_cursor = 0
        self._batch_agents: np.ndarray | None = None
        self._start = 0
        self._maximum = 0
        self._state: torch.Tensor | None = None
        self._previous_prediction: torch.Tensor | None = None
        self._previous_valid: torch.Tensor | None = None
        self._new_cycle()

    def _new_cycle(self) -> None:
        order = self.rng.permutation(self.agents)
        self._batches = [
            batch.copy() for batch in _agent_batches(order, self.batch_size)
        ]
        self._batch_cursor = 0
        self.cycles_started += 1

    def _new_batch(self, model: VQ2PhaseRecurrentActor) -> bool:
        if self._batch_cursor >= len(self._batches):
            if not self.cyclic:
                return False
            self._new_cycle()
        self._batch_agents = self._batches[self._batch_cursor]
        self._batch_cursor += 1
        self._start = 0
        self._maximum = int(self.dataset.lengths[self._batch_agents].max())
        self._state = model.initial_state(
            len(self._batch_agents), device=self.device
        )
        self._previous_prediction = None
        self._previous_valid = None
        return True

    def next(self, model: VQ2PhaseRecurrentActor) -> StreamItem | None:
        if self._batch_agents is None or self._start >= self._maximum:
            if not self._new_batch(model):
                return None
        assert self._batch_agents is not None and self._state is not None
        end = min(self._start + self.sequence_chunk, self._maximum)
        observation, target, valid, transition = self.dataset.chunk(
            self._batch_agents, self._start, end, device=self.device
        )
        item = StreamItem(
            observation=observation,
            target=target,
            valid=valid,
            transition=transition,
            start_state=self._state.detach(),
            previous_prediction=self._previous_prediction,
            previous_valid=self._previous_valid,
        )
        self._start = end
        self.source_chunks += 1
        self.source_records += int(valid.sum().item())
        return item

    def commit(
        self,
        output: torch.Tensor,
        next_state: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        self._state = next_state.detach()
        self._previous_prediction = output[:, -1].detach()
        self._previous_valid = valid[:, -1].detach()


def runtime_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "numpy": np.__version__,
    }


def source_paths() -> list[Path]:
    """Return the complete immutable VG010 source and evidence surface."""

    return [
        Path(__file__).resolve(),
        PREREGISTRATION,
        GOAL_PROMPT,
        DAGGER_ADMISSION,
        ROOT / "scripts/run_vq2_vg010_vast.sh",
        ROOT / "scripts/train_vq2_variable_gate_recurrent_bc.py",
        ROOT / "scripts/train_vq2_recurrent_bc.py",
        ROOT / "pufferlib/vq2_informed.py",
        ROOT / "pufferlib/vq2_recurrent.py",
        ROOT / "pufferlib/vq2_recurrent_phase.py",
        CLEAN_DATASET / "report.json",
        CLEAN_DATASET / "metadata.json",
        DAGGER_DATASET / "report.json",
        DAGGER_DATASET / "metadata.json",
        PARENT_CHECKPOINT,
        PARENT_REPORT,
    ]


def current_source_identity() -> tuple[str, dict[str, str]]:
    hashes = {source_label(path): sha256_path(path) for path in source_paths()}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, hashes


def verify_completed_output(output: Path, report: dict[str, Any]) -> None:
    """Fail closed when a completed-output resume is not source-identical."""

    if (
        report.get("schema")
        != "vq2_variable_gate_source_balanced_refit_report_v1"
        or report.get("tag") != TAG
        or not report.get("completed")
    ):
        raise RuntimeError("existing VG010 report identity changed")
    source_commit, source_sha256 = current_source_identity()
    if report.get("source_commit") != source_commit:
        raise RuntimeError("completed VG010 source commit changed")
    if report.get("source_sha256") != source_sha256:
        raise RuntimeError("completed VG010 source hashes changed")
    expected_evidence = {
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "clean_report_sha256": CLEAN_REPORT_SHA256,
        "clean_metadata_sha256": CLEAN_METADATA_SHA256,
        "dagger_report_sha256": DAGGER_REPORT_SHA256,
        "dagger_metadata_sha256": DAGGER_METADATA_SHA256,
    }
    for key, expected in expected_evidence.items():
        if report.get(key) != expected:
            raise RuntimeError(f"completed VG010 evidence changed for {key}")
    checkpoint_path = output / "policy_best.pt"
    if sha256_path(checkpoint_path) != report.get("checkpoint_sha256"):
        raise RuntimeError("completed VG010 checkpoint hash mismatch")
    state_path = output / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        state.get("schema") != "vq2_variable_gate_source_balanced_refit_state_v1"
        or state.get("tag") != TAG
        or state.get("status") != "completed"
        or state.get("source_commit") != source_commit
        or state.get("source_sha256") != source_sha256
        or state.get("checkpoint_sha256") != report.get("checkpoint_sha256")
        or state.get("report_sha256") != sha256_path(output / "report.json")
    ):
        raise RuntimeError("completed VG010 state does not bind the report")


def source_balanced_loss(
    clean_prediction: torch.Tensor,
    clean_target: torch.Tensor,
    clean_valid: torch.Tensor,
    dagger_prediction: torch.Tensor,
    dagger_target: torch.Tensor,
    dagger_valid: torch.Tensor,
    *,
    weights: torch.Tensor,
    clean_previous_prediction: torch.Tensor | None,
    clean_previous_valid: torch.Tensor | None,
    dagger_previous_prediction: torch.Tensor | None,
    dagger_previous_valid: torch.Tensor | None,
    config: RefitConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Assign each source exactly half of every optimizer objective."""

    clean_action, _ = weighted_action_mse(
        clean_prediction, clean_target, clean_valid, weights
    )
    dagger_action, _ = weighted_action_mse(
        dagger_prediction, dagger_target, dagger_valid, weights
    )
    clean_smooth = temporal_smoothness(
        clean_prediction,
        clean_valid,
        previous_prediction=clean_previous_prediction,
        previous_valid=clean_previous_valid,
    )
    dagger_smooth = temporal_smoothness(
        dagger_prediction,
        dagger_valid,
        previous_prediction=dagger_previous_prediction,
        previous_valid=dagger_previous_valid,
    )
    clean_total = clean_action + config.smoothness_weight * clean_smooth
    dagger_total = dagger_action + config.smoothness_weight * dagger_smooth
    total = (
        config.clean_objective_weight * clean_total
        + config.dagger_objective_weight * dagger_total
    )
    return total, {
        "clean_action": clean_action,
        "dagger_action": dagger_action,
        "clean_smoothness": clean_smooth,
        "dagger_smoothness": dagger_smooth,
        "clean_total": clean_total,
        "dagger_total": dagger_total,
    }


def _dataset_split(agents: int, validation_agents: int) -> tuple[np.ndarray, np.ndarray]:
    if validation_agents <= 0 or validation_agents >= agents:
        raise ValueError("validation split must leave source training agents")
    boundary = agents - validation_agents
    return np.arange(boundary, dtype=np.int64), np.arange(
        boundary, agents, dtype=np.int64
    )


def _load_parent(model: VQ2PhaseRecurrentActor) -> dict[str, Any]:
    if sha256_path(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("VG005 parent checkpoint hash mismatch")
    if sha256_path(PARENT_REPORT) != PARENT_REPORT_SHA256:
        raise RuntimeError("VG005 parent report hash mismatch")
    report = json.loads(PARENT_REPORT.read_text())
    if not report.get("completed"):
        raise RuntimeError("VG005 parent report is not complete")
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    contract = payload.get("model", {})
    if (
        payload.get("schema") != SCHEMA
        or contract.get("legal_observation_size") != PHASE_LEGAL_OBS_SIZE
        or contract.get("hidden_size") != model.hidden_size
    ):
        raise RuntimeError("VG005 parent actor contract changed")
    model.load_state_dict(payload["model_state"])
    return payload


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
    baseline_validation: dict[str, Any],
    best_epoch: int,
    best_score: float,
    best_state: dict[str, torch.Tensor],
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
        "baseline_validation": baseline_validation,
        "best_epoch": best_epoch,
        "best_source_balanced_validation": best_score,
        "best_state": best_state,
        "optimizer_updates": optimizer_updates,
    }


def train(
    *,
    output: Path = DEFAULT_OUTPUT,
    device_name: str = "cuda",
    config: RefitConfig = RefitConfig(),
    resume: bool = False,
) -> dict[str, Any]:
    if config.sequence_chunk < 256:
        raise RuntimeError("VG010 requires BPTT windows of at least 256 steps")
    if config.transition_window_exposure < 3:
        raise RuntimeError("VG010 requires at least 3x transition-window exposure")
    if abs(config.clean_objective_weight - 0.5) > 1e-12 or abs(
        config.dagger_objective_weight - 0.5
    ) > 1e-12:
        raise RuntimeError("VG010 requires exact equal source objective weight")
    if sha256_path(GOAL_PROMPT) != GOAL_PROMPT_SHA256:
        raise RuntimeError("persistent goal prompt hash mismatch")
    if sha256_path(DAGGER_ADMISSION) != DAGGER_ADMISSION_SHA256:
        raise RuntimeError("VG009 admission evidence hash mismatch")
    if not PREREGISTRATION.is_file():
        raise RuntimeError("VG010 preregistration is missing")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VG010 preregisters CUDA training")

    report_path = output / "report.json"
    state_path = output / "training_state.pt"
    if report_path.is_file():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {output}")
        report = json.loads(report_path.read_text())
        verify_completed_output(output, report)
        return report
    if output.exists() and not state_path.is_file():
        raise RuntimeError("VG010 output exists without resumable state")

    device = torch.device(device_name)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    clean = VariableGateBCDataset(
        CLEAN_DATASET,
        verify_hashes=True,
        expected_report_sha256=CLEAN_REPORT_SHA256,
        expected_metadata_sha256=CLEAN_METADATA_SHA256,
    )
    dagger = VariableGateBCDataset(
        DAGGER_DATASET,
        verify_hashes=True,
        expected_report_sha256=DAGGER_REPORT_SHA256,
        expected_metadata_sha256=DAGGER_METADATA_SHA256,
    )
    clean_train, clean_validation = _dataset_split(
        clean.agents, config.clean_validation_agents
    )
    dagger_train, dagger_validation = _dataset_split(
        dagger.agents, config.dagger_validation_agents
    )

    model = VQ2PhaseRecurrentActor(
        hidden_size=config.hidden_size,
        initial_std=config.initial_std,
    ).to(device)
    parent = _load_parent(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(config.seed)

    source_commit, source_sha256 = current_source_identity()
    identity = {
        "schema": "vq2_variable_gate_source_balanced_refit_state_v1",
        "tag": TAG,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": runtime_manifest(),
        "train_config": asdict(config),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "clean_report_sha256": CLEAN_REPORT_SHA256,
        "clean_metadata_sha256": CLEAN_METADATA_SHA256,
        "dagger_report_sha256": DAGGER_REPORT_SHA256,
        "dagger_metadata_sha256": DAGGER_METADATA_SHA256,
        "clean_phase_audit": clean.phase_audit,
        "dagger_phase_audit": dagger.phase_audit,
        "source_balancing": {
            "per_optimizer_step": {"clean": 0.5, "dagger": 0.5},
            "raw_record_count_does_not_set_source_weight": True,
            "clean_training_agents": len(clean_train),
            "dagger_training_agents": len(dagger_train),
            "clean_validation_agents": len(clean_validation),
            "dagger_validation_agents": len(dagger_validation),
        },
        "safety": {
            "actor_input_privileged_values": 0,
            "teacher_blend": 0.0,
            "flight_sim_packets_sent": 0,
            "sealed_test_accesses": 0,
            "submission_authorized": False,
        },
    }

    baseline_clean = evaluate(model, clean, clean_validation, config, device)
    baseline_dagger = evaluate(model, dagger, dagger_validation, config, device)
    baseline_validation = {
        "clean": baseline_clean,
        "dagger": baseline_dagger,
        "source_balanced_weighted_mse": 0.5
        * (baseline_clean["weighted_mse"] + baseline_dagger["weighted_mse"]),
    }
    best_epoch = 0
    best_score = float(baseline_validation["source_balanced_weighted_mse"])
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    history: list[dict[str, Any]] = []
    optimizer_updates = 0
    completed_epoch = 0

    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"VG010 state already exists at {state_path}")
        saved = torch.load(state_path, map_location=device, weights_only=False)
        for key, expected in identity.items():
            if saved.get(key) != expected:
                raise RuntimeError(f"VG010 resume mismatch for {key}")
        if saved.get("status") != "training":
            raise RuntimeError("VG010 can resume only active training state")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        random.setstate(saved["python_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        history = saved["history"]
        baseline_validation = saved["baseline_validation"]
        best_epoch = int(saved["best_epoch"])
        best_score = float(saved["best_source_balanced_validation"])
        best_state = saved["best_state"]
        optimizer_updates = int(saved["optimizer_updates"])
        completed_epoch = int(saved["completed_epoch"])
    else:
        if resume:
            raise RuntimeError("VG010 --resume requested without state")
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
                baseline_validation=baseline_validation,
                best_epoch=best_epoch,
                best_score=best_score,
                best_state=best_state,
                optimizer_updates=optimizer_updates,
            ),
        )

    weights = torch.tensor(config.action_weights, device=device)
    started = time.perf_counter()
    for epoch in range(completed_epoch + 1, config.epochs + 1):
        model.train()
        epoch_optimizer_updates_start = optimizer_updates
        clean_stream = RecurrentChunkStream(
            clean,
            clean_train,
            batch_size=config.source_agent_batch_size,
            sequence_chunk=config.sequence_chunk,
            rng=rng,
            device=device,
            cyclic=False,
        )
        dagger_stream = RecurrentChunkStream(
            dagger,
            dagger_train,
            batch_size=config.source_agent_batch_size,
            sequence_chunk=config.sequence_chunk,
            rng=rng,
            device=device,
            cyclic=True,
        )
        clean_metrics = ActionAccumulator()
        dagger_metrics = ActionAccumulator()
        paired_updates = 0
        transition_paired_updates = 0
        transition_update_exposures = 0
        clean_weight_sum = 0.0
        dagger_weight_sum = 0.0
        total_loss_sum = 0.0
        clean_loss_sum = 0.0
        dagger_loss_sum = 0.0
        while True:
            clean_item = clean_stream.next(model)
            if clean_item is None:
                break
            dagger_item = dagger_stream.next(model)
            if dagger_item is None:
                raise RuntimeError("VG010 cyclic DAgger stream ended")
            transition_present = bool(
                clean_item.transition.any() or dagger_item.transition.any()
            )
            repetitions = (
                config.transition_window_exposure if transition_present else 1
            )
            if transition_present:
                transition_paired_updates += 1
            final_clean_output = None
            final_clean_state = None
            final_dagger_output = None
            final_dagger_state = None
            final_components: dict[str, torch.Tensor] | None = None
            for _ in range(repetitions):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    clean_output, clean_state = model.forward_sequence(
                        clean_item.observation, clean_item.start_state
                    )
                    dagger_output, dagger_state = model.forward_sequence(
                        dagger_item.observation, dagger_item.start_state
                    )
                    loss, components = source_balanced_loss(
                        clean_output.mean,
                        clean_item.target,
                        clean_item.valid,
                        dagger_output.mean,
                        dagger_item.target,
                        dagger_item.valid,
                        weights=weights,
                        clean_previous_prediction=clean_item.previous_prediction,
                        clean_previous_valid=clean_item.previous_valid,
                        dagger_previous_prediction=dagger_item.previous_prediction,
                        dagger_previous_valid=dagger_item.previous_valid,
                        config=config,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer_updates += 1
                transition_update_exposures += int(transition_present)
                clean_weight_sum += config.clean_objective_weight
                dagger_weight_sum += config.dagger_objective_weight
                total_loss_sum += float(loss.detach().item())
                clean_loss_sum += float(components["clean_total"].detach().item())
                dagger_loss_sum += float(components["dagger_total"].detach().item())
                final_clean_output = clean_output
                final_clean_state = clean_state
                final_dagger_output = dagger_output
                final_dagger_state = dagger_state
                final_components = components
            if (
                final_clean_output is None
                or final_clean_state is None
                or final_dagger_output is None
                or final_dagger_state is None
                or final_components is None
            ):
                raise RuntimeError("VG010 emitted no paired update")
            clean_metrics.add(
                final_clean_output.mean, clean_item.target, clean_item.valid
            )
            dagger_metrics.add(
                final_dagger_output.mean, dagger_item.target, dagger_item.valid
            )
            clean_stream.commit(
                final_clean_output.mean, final_clean_state, clean_item.valid
            )
            dagger_stream.commit(
                final_dagger_output.mean, final_dagger_state, dagger_item.valid
            )
            paired_updates += 1

        clean_validation_result = evaluate(
            model, clean, clean_validation, config, device
        )
        dagger_validation_result = evaluate(
            model, dagger, dagger_validation, config, device
        )
        validation_score = 0.5 * (
            clean_validation_result["weighted_mse"]
            + dagger_validation_result["weighted_mse"]
        )
        if abs(clean_weight_sum - dagger_weight_sum) > 1e-9:
            raise RuntimeError("VG010 source objective weights diverged")
        minimum_transition_exposure = (
            transition_update_exposures / transition_paired_updates
            if transition_paired_updates
            else 0.0
        )
        epoch_report = {
            "epoch": epoch,
            "paired_chunks": paired_updates,
            "optimizer_updates": optimizer_updates,
            "epoch_optimizer_updates": optimizer_updates
            - epoch_optimizer_updates_start,
            "transition_paired_chunks": transition_paired_updates,
            "transition_update_exposures": transition_update_exposures,
            "minimum_transition_window_exposure": minimum_transition_exposure,
            "clean_objective_weight_sum": clean_weight_sum,
            "dagger_objective_weight_sum": dagger_weight_sum,
            "source_weight_difference": clean_weight_sum - dagger_weight_sum,
            "clean_stream_chunks": clean_stream.source_chunks,
            "dagger_stream_chunks": dagger_stream.source_chunks,
            "clean_stream_records": clean_stream.source_records,
            "dagger_stream_records": dagger_stream.source_records,
            "dagger_cycles_started": dagger_stream.cycles_started,
            "mean_total_loss_per_update": total_loss_sum
            / max(clean_weight_sum + dagger_weight_sum, 1.0),
            "mean_clean_loss_per_update": clean_loss_sum
            / max(2.0 * clean_weight_sum, 1.0),
            "mean_dagger_loss_per_update": dagger_loss_sum
            / max(2.0 * dagger_weight_sum, 1.0),
            "train_clean": clean_metrics.result(),
            "train_dagger": dagger_metrics.result(),
            "validation_clean": clean_validation_result,
            "validation_dagger": dagger_validation_result,
            "source_balanced_validation_weighted_mse": validation_score,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        if np.isfinite(validation_score) and validation_score < best_score:
            best_score = float(validation_score)
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
                baseline_validation=baseline_validation,
                best_epoch=best_epoch,
                best_score=best_score,
                best_state=best_state,
                optimizer_updates=optimizer_updates,
            ),
        )

    minimum_transition_exposure = min(
        float(epoch["minimum_transition_window_exposure"]) for epoch in history
    )
    equal_weight_audit = all(
        abs(float(epoch["source_weight_difference"])) <= 1e-9 for epoch in history
    )
    if minimum_transition_exposure + 1e-12 < config.transition_window_exposure:
        raise RuntimeError("VG010 transition exposure contract failed")
    if not equal_weight_audit:
        raise RuntimeError("VG010 source balance audit failed")

    selected_clean = (
        baseline_validation["clean"]
        if best_epoch == 0
        else history[best_epoch - 1]["validation_clean"]
    )
    selected_dagger = (
        baseline_validation["dagger"]
        if best_epoch == 0
        else history[best_epoch - 1]["validation_dagger"]
    )
    numerically_admitted = bool(
        best_epoch > 0
        and np.isfinite(best_score)
        and best_score
        < float(baseline_validation["source_balanced_weighted_mse"])
        and selected_dagger["weighted_mse"]
        < baseline_validation["dagger"]["weighted_mse"]
        and selected_clean["weighted_mse"]
        <= config.maximum_clean_validation_weighted_mse
        and equal_weight_audit
    )
    checkpoint = {
        "schema": SCHEMA,
        "tag": TAG,
        "model": parent["model"],
        "model_state": best_state,
        "train_config": asdict(config),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "clean_train_agents": clean_train.tolist(),
        "clean_validation_agents": clean_validation.tolist(),
        "dagger_train_agents": dagger_train.tolist(),
        "dagger_validation_agents": dagger_validation.tolist(),
        "clean_report_sha256": CLEAN_REPORT_SHA256,
        "clean_metadata_sha256": CLEAN_METADATA_SHA256,
        "dagger_report_sha256": DAGGER_REPORT_SHA256,
        "dagger_metadata_sha256": DAGGER_METADATA_SHA256,
        "clean_phase_audit": clean.phase_audit,
        "dagger_phase_audit": dagger.phase_audit,
        "baseline_validation": baseline_validation,
        "best_epoch": best_epoch,
        "best_source_balanced_validation": best_score,
        "selected_validation_clean": selected_clean,
        "selected_validation_dagger": selected_dagger,
        "history": history,
        "optimizer_updates": optimizer_updates,
        "minimum_transition_window_exposure": minimum_transition_exposure,
        "equal_source_weight_audit": equal_weight_audit,
        "numerically_admitted": numerically_admitted,
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "runtime": identity["runtime"],
        "safety": identity["safety"],
    }
    checkpoint_path = output / "policy_best.pt"
    atomic_torch_save(checkpoint_path, checkpoint)
    report = {
        "schema": "vq2_variable_gate_source_balanced_refit_report_v1",
        "tag": TAG,
        "completed": True,
        "numerically_admitted": numerically_admitted,
        "checkpoint": source_label(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "best_epoch": best_epoch,
        "best_source_balanced_validation": best_score,
        "baseline_validation": baseline_validation,
        "selected_validation_clean": selected_clean,
        "selected_validation_dagger": selected_dagger,
        "minimum_transition_window_exposure": minimum_transition_exposure,
        "equal_source_weight_audit": equal_weight_audit,
        "optimizer_updates": optimizer_updates,
        "history": history,
        "train_config": asdict(config),
        "clean_report_sha256": CLEAN_REPORT_SHA256,
        "clean_metadata_sha256": CLEAN_METADATA_SHA256,
        "dagger_report_sha256": DAGGER_REPORT_SHA256,
        "dagger_metadata_sha256": DAGGER_METADATA_SHA256,
        "clean_phase_audit": clean.phase_audit,
        "dagger_phase_audit": dagger.phase_audit,
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
        baseline_validation=baseline_validation,
        best_epoch=best_epoch,
        best_score=best_score,
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
        output=args.output.resolve(), device_name=args.device, resume=args.resume
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["numerically_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
