import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_teacher_intervention_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("teacher_intervention_dataset", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_build_records_uses_next_observation_executed_action_and_drops_reset():
    observations = np.zeros((5, 2, 32), dtype=np.float32)
    terminals = np.zeros((5, 2), dtype=np.float32)
    terminals[4, 0] = 1.0
    terminals[3, 1] = 1.0
    for time in range(1, 5):
        observations[time, :, 19:23] = time * np.ones(4, dtype=np.float32)

    records, summary = module.build_records(observations, terminals)

    assert summary == {
        "episodes": 2,
        "records": 5,
        "dropped_unobservable_final_actions": 2,
    }
    assert records[:, -1].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert records[:3, 32:36].tolist() == [[1.0] * 4, [2.0] * 4, [3.0] * 4]
    assert records[3:, 32:36].tolist() == [[1.0] * 4, [2.0] * 4]


def test_build_records_rejects_mismatched_terminal_shape():
    with pytest.raises(ValueError, match="terminal shape"):
        module.build_records(
            np.zeros((2, 3, 32), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )


def test_build_records_can_keep_only_terminal_reward_successes():
    observations = np.zeros((5, 3, 32), dtype=np.float32)
    terminals = np.zeros((5, 3), dtype=np.float32)
    rewards = np.zeros((5, 3), dtype=np.float32)
    terminals[3, 0] = 1.0
    terminals[4, 1] = 1.0
    rewards[3, 0] = 180.0
    rewards[4, 1] = -600.0
    for time in range(1, 5):
        observations[time, :, 19:23] = float(time)

    records, summary = module.build_records(
        observations,
        terminals,
        rewards=rewards,
        success_reward_threshold=100.0,
    )

    assert summary == {
        "episodes": 1,
        "records": 2,
        "dropped_unobservable_final_actions": 1,
        "rejected_unsuccessful_episodes": 2,
    }
    assert records[:, -1].tolist() == [1.0, 0.0]
    assert records[:, 32:36].tolist() == [[1.0] * 4, [2.0] * 4]


def test_build_records_requires_rewards_for_success_filter():
    with pytest.raises(ValueError, match="rewards are required"):
        module.build_records(
            np.zeros((2, 1, 32), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            success_reward_threshold=100.0,
        )
