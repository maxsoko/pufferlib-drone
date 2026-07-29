from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.train_vq2_variable_gate_recurrent_bc as trainer
from pufferlib.vq2_informed import MASK_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.train_vq2_variable_gate_recurrent_bc import (
    PHASE_TAIL_INDEX,
    TrainConfig,
    VariableGateBCDataset,
    actor_agent_split,
    phase_increment_rows,
    reconstruct_phase_batch,
    sha256_path,
)


def test_vg004_contract_uses_256_step_bptt_and_3x_transition_exposure() -> None:
    config = TrainConfig()
    assert PHASE_LEGAL_OBS_SIZE == 4119
    assert config.hidden_size == 256
    assert config.sequence_chunk >= 256
    assert config.transition_window_exposure >= 3
    assert config.validation_agents == 32
    assert config.epochs == 12


def test_reconstruct_phase_batch_is_batch_major_and_exact() -> None:
    mask = np.zeros((3, 2, MASK_SIZE), dtype=np.uint8)
    mask[1, 0, 7] = 255
    tail = np.zeros(
        (3, 2, PHASE_LEGAL_OBS_SIZE - MASK_SIZE), dtype=np.float32
    )
    tail[2, 1, -1] = 7.0 / 16.0
    observation = reconstruct_phase_batch(mask, tail, device=torch.device("cpu"))
    assert observation.shape == (2, 3, PHASE_LEGAL_OBS_SIZE)
    assert observation[0, 1, 7].item() == 1.0
    assert observation[1, 2, -1].item() == pytest.approx(7.0 / 16.0)


def test_phase_increment_rows_preserves_chunk_boundary_transition() -> None:
    phase = np.asarray(
        [[0.0, 0.0], [1.0 / 16.0, 0.0], [1.0 / 16.0, 2.0 / 16.0]],
        dtype=np.float32,
    )
    valid = np.ones_like(phase, dtype=np.uint8)
    increments = phase_increment_rows(
        phase, valid, np.zeros(2, dtype=np.float32)
    )
    assert increments.tolist() == [[False, False], [True, False], [False, True]]

    second = phase_increment_rows(
        np.asarray([[2.0 / 16.0, 2.0 / 16.0]], dtype=np.float32),
        np.ones((1, 2), dtype=np.uint8),
        phase[-1],
    )
    assert second.tolist() == [[True, False]]


def test_phase_increment_rows_rejects_decrease_or_misalignment() -> None:
    with pytest.raises(RuntimeError, match="decreased"):
        phase_increment_rows(
            np.asarray([[1.0 / 16.0], [0.0]], dtype=np.float32),
            np.ones((2, 1), dtype=np.uint8),
            np.zeros(1, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="previous phase"):
        phase_increment_rows(
            np.zeros((2, 1), dtype=np.float32),
            np.ones((2, 1), dtype=np.uint8),
            np.zeros(2, dtype=np.float32),
        )


def test_agent_split_reserves_final_exact_uniform_block() -> None:
    train, validation = actor_agent_split(256, 32)
    assert train.tolist() == list(range(224))
    assert validation.tolist() == list(range(224, 256))
    assert not np.intersect1d(train, validation).size


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_dataset_chunk_reconstructs_phase_and_finds_boundary_increment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    steps, agents = 4, 2
    mask = np.zeros((steps, agents, MASK_SIZE), dtype=np.uint8)
    tail = np.zeros(
        (steps, agents, PHASE_LEGAL_OBS_SIZE - MASK_SIZE), dtype=np.float32
    )
    tail[:, 0, PHASE_TAIL_INDEX] = [0.0, 1.0 / 16.0, 1.0 / 16.0, 2.0 / 16.0]
    tail[:, 1, PHASE_TAIL_INDEX] = [0.0, 0.0, 1.0 / 16.0, 1.0 / 16.0]
    action = np.zeros((steps, agents, ACTION_SIZE), dtype=np.float32)
    valid = np.ones((steps, agents), dtype=np.uint8)
    terminal = np.zeros((steps, agents), dtype=np.uint8)
    terminal[-1] = 1
    np.save(root / "mask.npy", mask)
    np.save(root / "tail.npy", tail)
    np.save(root / "action.npy", action)
    np.save(root / "valid.npy", valid)
    np.save(root / "terminal.npy", terminal)
    metadata = {
        "episode_lengths": [4, 4],
        "observation": {
            "stored_legal_width": PHASE_LEGAL_OBS_SIZE,
            "mask_width": MASK_SIZE,
            "legal_sensor_history_width": 22,
            "public_phase_width": 1,
            "public_phase_tail_index": 22,
            "public_phase_encoding": "clamp(active_gate_index,0,16)/16",
            "public_phase_rate_hz": 4,
            "stored_training_only_privileged_values_per_record": 0,
            "stored_total_gate_count_values_per_record": 0,
        },
        "files": {},
    }
    report = {"admitted": True}
    _write_json(root / "metadata.json", metadata)
    _write_json(root / "report.json", report)
    dataset = VariableGateBCDataset(
        root,
        verify_hashes=False,
        expected_report_sha256=sha256_path(root / "report.json"),
        expected_metadata_sha256=sha256_path(root / "metadata.json"),
    )
    observation, targets, selected, transition = dataset.chunk(
        np.asarray([0, 1]), 2, 4, device=torch.device("cpu")
    )
    assert observation.shape == (2, 2, PHASE_LEGAL_OBS_SIZE)
    assert targets.shape == (2, 2, ACTION_SIZE)
    assert selected.all()
    assert transition.tolist() == [[False, True], [True, False]]

    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text("source-locked test\n")
    monkeypatch.setattr(trainer, "DATASET", root)
    monkeypatch.setattr(
        trainer, "DATASET_REPORT_SHA256", sha256_path(root / "report.json")
    )
    monkeypatch.setattr(
        trainer,
        "DATASET_METADATA_SHA256",
        sha256_path(root / "metadata.json"),
    )
    monkeypatch.setattr(trainer, "PREREGISTRATION", preregistration)
    config = TrainConfig(
        hidden_size=16,
        epochs=1,
        agent_batch_size=1,
        sequence_chunk=256,
        validation_agents=1,
        transition_window_exposure=3,
    )
    training_report = trainer.train(
        output=tmp_path / "training",
        device_name="cpu",
        config=config,
    )
    assert training_report["completed"]
    assert training_report["minimum_transition_window_exposure"] == 3.0
    assert training_report["best_epoch"] == 1
