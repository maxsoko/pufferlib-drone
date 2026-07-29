from pathlib import Path

import numpy as np
import pytest
import torch

from pufferlib.vq2_informed import ENV_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_boundary_plane_capacity import (
    BOUNDARY_DATASET_FIELDS,
    _boundary_state,
    _load_boundary_event_dataset,
)


def _boundary_dataset(events: int = 2, context: int = 3) -> dict[str, np.ndarray]:
    length = context + 1
    data = {
        "mask": np.zeros((events, length, MASK_SIZE), dtype=np.uint8),
        "tail": np.zeros(
            (events, length, ENV_OBS_SIZE - MASK_SIZE), dtype=np.float16
        ),
        "action": np.zeros((events, length, 4), dtype=np.float16),
        "reward": np.zeros((events, length), dtype=np.float16),
        "continuation": np.ones((events, length), dtype=np.uint8),
        "agent": np.arange(events, dtype=np.int16),
        "vector_step": np.arange(events, dtype=np.int32),
        "phase_after_event": np.ones(events, dtype=np.float16) / 6.0,
        "initial_deterministic": np.zeros((events, 5), dtype=np.float16),
        "initial_stochastic_index": np.zeros((events, 2), dtype=np.uint8),
        "initial_logits": np.zeros((events, 2, 3), dtype=np.float16),
        "preevent_deterministic": np.zeros((events, 5), dtype=np.float16),
        "preevent_stochastic_index": np.zeros((events, 2), dtype=np.uint8),
        "preevent_logits": np.zeros((events, 2, 3), dtype=np.float16),
    }
    data["reward"][:, -1] = 30.0
    assert set(data) == BOUNDARY_DATASET_FIELDS
    return data


def test_boundary_loader_validates_exact_v2_schema(tmp_path: Path) -> None:
    path = tmp_path / "events.npz"
    np.savez_compressed(path, **_boundary_dataset())
    loaded = _load_boundary_event_dataset(
        path,
        context_length=3,
        event_threshold=1.0,
        deterministic_size=5,
        stochastic_groups=2,
        stochastic_classes=3,
    )
    assert loaded["initial_deterministic"].shape == (2, 5)

    invalid = _boundary_dataset()
    invalid.pop("preevent_logits")
    invalid_path = tmp_path / "invalid.npz"
    np.savez_compressed(invalid_path, **invalid)
    with pytest.raises(RuntimeError, match="schema mismatch"):
        _load_boundary_event_dataset(
            invalid_path,
            context_length=3,
            event_threshold=1.0,
            deterministic_size=5,
            stochastic_groups=2,
            stochastic_classes=3,
        )


def test_boundary_state_reconstructs_hard_categories() -> None:
    dataset = _boundary_dataset(events=2)
    dataset["initial_stochastic_index"][:] = np.asarray([[0, 2], [1, 0]])
    state = _boundary_state(
        dataset, device=torch.device("cpu"), stochastic_classes=3
    )
    assert state.deterministic.shape == (2, 5)
    assert state.stochastic.shape == (2, 2, 3)
    assert state.stochastic.argmax(-1).tolist() == [[0, 2], [1, 0]]
    assert torch.equal(state.stochastic.sum(-1), torch.ones(2, 2))


def test_boundary_loader_rejects_early_event(tmp_path: Path) -> None:
    dataset = _boundary_dataset()
    dataset["reward"][:, -2] = 30.0
    path = tmp_path / "early.npz"
    np.savez_compressed(path, **dataset)
    with pytest.raises(RuntimeError, match="early event"):
        _load_boundary_event_dataset(
            path,
            context_length=3,
            event_threshold=1.0,
            deterministic_size=5,
            stochastic_groups=2,
            stochastic_classes=3,
        )
