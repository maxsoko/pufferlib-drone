from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.continue_vq2_event_balanced_world_offline import (
    _event_batch,
    _load_event_dataset,
    _mix_batch,
    _resolve_world_learning_rate,
)


def _write_event_fixture(path: Path) -> None:
    events = 2
    length = 3
    np.savez_compressed(
        path,
        mask=np.zeros((events, length, 4096), dtype=np.uint8),
        tail=np.zeros((events, length, 56), dtype=np.float16),
        action=np.zeros((events, length, 4), dtype=np.float16),
        reward=np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=np.float16),
        continuation=np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.uint8),
        agent=np.asarray([0, 1], dtype=np.int16),
        vector_step=np.asarray([10, 20], dtype=np.int32),
        phase_after_event=np.asarray([1 / 6, 2 / 6], dtype=np.float16),
    )


def test_load_and_decode_event_dataset(tmp_path: Path) -> None:
    path = tmp_path / "events.npz"
    _write_event_fixture(path)
    dataset = _load_event_dataset(path, context_length=2, event_threshold=1.0)
    observation, action, reward, continuation = _event_batch(
        dataset, [1], device=torch.device("cpu")
    )
    assert observation.shape == (1, 3, 4152)
    assert action.shape == (1, 3, 4)
    assert reward[0, -1].item() == 3.0
    assert continuation[0, -1].item() == 0.0


def test_load_event_dataset_rejects_invalid_context(tmp_path: Path) -> None:
    path = tmp_path / "events.npz"
    _write_event_fixture(path)
    with np.load(path) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    values["continuation"][0, 0] = 0
    np.savez_compressed(path, **values)
    with pytest.raises(RuntimeError, match="pre-event terminal"):
        _load_event_dataset(path, context_length=2, event_threshold=1.0)


def test_mix_batch_applies_locked_permutation() -> None:
    dense = tuple(torch.full((2, 1), float(index)) for index in range(4))
    event = tuple(torch.full((1, 1), float(index + 10)) for index in range(4))
    mixed = _mix_batch(dense, event, permutation=torch.tensor([2, 0, 1]))
    assert mixed[0].flatten().tolist() == [0.0, 10.0, 0.0]
    assert mixed[3].flatten().tolist() == [3.0, 13.0, 3.0]


def test_world_learning_rate_override_requires_reset() -> None:
    assert _resolve_world_learning_rate(4e-5, 0.0, reset_optimizer=False) == 4e-5
    assert _resolve_world_learning_rate(4e-5, 3e-5, reset_optimizer=True) == 3e-5
    with pytest.raises(ValueError, match="fresh optimizer"):
        _resolve_world_learning_rate(4e-5, 3e-5, reset_optimizer=False)
