from __future__ import annotations

import numpy as np
import torch

import scripts.train_vq2_lc065_phase2_failure_direction as lc065


def test_terminal_outcome_uses_each_agents_last_phase2_record() -> None:
    agents = np.array([0, 1, 0, 1, 2, 2], dtype=np.int64)
    steps = np.array([4, 4, 5, 5, 7, 8], dtype=np.int64)
    terminals = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    present, outcome = lc065.terminal_outcome_by_agent(
        agents, steps, terminals, capacity=4
    )
    assert present.tolist() == [0, 1, 2]
    assert outcome.tolist() == [1, 0, 1, -1]


def test_failure_direction_normalizes_agent_class_means() -> None:
    features = torch.tensor([[0.0], [0.2], [1.0], [1.2]])
    agents = np.array([0, 0, 1, 1], dtype=np.int64)
    present = np.array([0, 1], dtype=np.int64)
    outcome = np.array([1, 0], dtype=np.int8)
    weight, bias, _, _ = lc065.normalize_failure_direction(
        torch.tensor([1.0]), torch.tensor(0.0),
        features, agents, present, outcome,
    )
    scores = (features @ weight + bias).numpy()
    means = lc065.agent_means(scores, agents, present)
    assert np.allclose(means, [0.0, 1.0], atol=1e-6)
