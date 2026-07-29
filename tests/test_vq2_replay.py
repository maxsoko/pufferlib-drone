from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from pufferlib.vq2_informed import ENV_OBS_SIZE, MASK_SIZE
from scripts.train_vq2_informed_dreamer import QuantizedSequenceReplay


def _add(
    replay: QuantizedSequenceReplay,
    value: float,
    *,
    continuation: bool = True,
    valid: bool = True,
) -> None:
    observation = torch.zeros(replay.agents, ENV_OBS_SIZE)
    observation[:, :MASK_SIZE] = value / 10.0
    observation[:, MASK_SIZE] = value
    action = torch.full((replay.agents, 4), value)
    reward = torch.full((replay.agents,), value)
    replay.add(
        observation,
        action,
        reward,
        torch.full((replay.agents,), float(continuation)),
        torch.full((replay.agents,), float(valid)),
    )


def test_replay_allows_terminal_only_at_final_sequence_element() -> None:
    replay = QuantizedSequenceReplay(capacity=8, agents=1)
    _add(replay, 1)
    _add(replay, 2)
    _add(replay, 3, continuation=False)
    _add(replay, 4, valid=False)  # deferred reset-only native step
    _add(replay, 5)
    _add(replay, 6)

    assert replay._candidate_starts(3).tolist() == [[0, 0]]
    observation, action, reward, continuation = replay.sample(
        1, 3, device=torch.device("cpu"), rng=random.Random(7)
    )
    assert observation[0, :, MASK_SIZE].tolist() == [1.0, 2.0, 3.0]
    assert action[0, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert reward[0].tolist() == [1.0, 2.0, 3.0]
    assert continuation[0].tolist() == [1.0, 1.0, 0.0]


def test_replay_candidates_remain_chronological_after_ring_wrap() -> None:
    replay = QuantizedSequenceReplay(capacity=5, agents=1)
    for value in range(1, 8):
        _add(replay, float(value))
    assert replay.oldest == 2
    assert replay._candidate_starts(3).tolist() == [[0, 0], [1, 0], [2, 0]]
    observation, *_ = replay.sample(
        1, 3, device=torch.device("cpu"), rng=random.Random(2)
    )
    assert observation[0, :, MASK_SIZE].tolist() in (
        [5.0, 6.0, 7.0],
        [4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0],
    )


def test_replay_rejects_when_no_reset_free_sequence_exists() -> None:
    replay = QuantizedSequenceReplay(capacity=4, agents=1)
    _add(replay, 1, continuation=False)
    _add(replay, 2, valid=False)
    _add(replay, 3)
    with pytest.raises(RuntimeError, match="reset-free sequence"):
        replay.sample(1, 3, device=torch.device("cpu"), rng=random.Random(1))


def test_mask_quantization_round_trip_is_bounded() -> None:
    replay = QuantizedSequenceReplay(capacity=2, agents=1)
    observation = torch.zeros(1, ENV_OBS_SIZE)
    observation[:, :MASK_SIZE] = torch.linspace(0.0, 1.0, MASK_SIZE)
    replay.add(
        observation,
        torch.zeros(1, 4),
        torch.zeros(1),
        torch.ones(1),
        torch.ones(1),
    )
    sampled, *_ = replay.sample(
        1, 1, device=torch.device("cpu"), rng=random.Random(0)
    )
    error = np.max(
        np.abs(sampled[0, 0, :MASK_SIZE].numpy() - observation[0, :MASK_SIZE].numpy())
    )
    assert error <= (0.5 / 255.0 + 1e-7)


def test_replay_reports_exact_storage_budget() -> None:
    replay = QuantizedSequenceReplay(capacity=3, agents=2)
    expected = sum(
        array.nbytes
        for array in (
            replay.mask,
            replay.tail,
            replay.action,
            replay.reward,
            replay.continuation,
            replay.valid,
        )
    )
    assert replay.storage_bytes == expected


def test_replay_sample_reports_age_and_recent_session_provenance() -> None:
    replay = QuantizedSequenceReplay(capacity=8, agents=1)
    for value in range(1, 7):
        _add(replay, float(value))
    (_observation, _action, _reward, _continuation), info = replay.sample(
        4,
        3,
        device=torch.device("cpu"),
        rng=random.Random(5),
        include_info=True,
        recent_steps=3,
    )
    assert info["newest_age_steps"].shape == (4,)
    assert np.array_equal(
        info["oldest_age_steps"], info["newest_age_steps"] + 2
    )
    assert np.array_equal(info["fully_recent"], info["newest_age_steps"] == 0)
    assert np.all(info["newest_age_steps"] >= 0)


def test_memory_mapped_replay_round_trips_and_resumes(tmp_path) -> None:
    directory = tmp_path / "replay"
    replay = QuantizedSequenceReplay(
        capacity=4, agents=1, storage_dir=directory
    )
    _add(replay, 1)
    _add(replay, 2)
    _add(replay, 3)
    replay.flush()

    restored = QuantizedSequenceReplay(
        capacity=4, agents=1, storage_dir=directory, resume=True
    )
    assert restored.position == 3
    assert restored.size == 3
    assert restored.state_dict()["schema"] == QuantizedSequenceReplay.SCHEMA
    observation, action, reward, continuation = restored.sample(
        1, 3, device=torch.device("cpu"), rng=random.Random(0)
    )
    assert observation[0, :, MASK_SIZE].tolist() == [1.0, 2.0, 3.0]
    assert action[0, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert reward[0].tolist() == [1.0, 2.0, 3.0]
    assert continuation[0].tolist() == [1.0, 1.0, 1.0]


def test_memory_mapped_replay_refuses_overwrite_and_bad_resume(tmp_path) -> None:
    directory = tmp_path / "replay"
    replay = QuantizedSequenceReplay(
        capacity=2, agents=1, storage_dir=directory
    )
    replay.flush()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        QuantizedSequenceReplay(capacity=2, agents=1, storage_dir=directory)
    with pytest.raises(RuntimeError, match="capacity mismatch"):
        QuantizedSequenceReplay(
            capacity=3, agents=1, storage_dir=directory, resume=True
        )
