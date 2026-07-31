from __future__ import annotations

import numpy as np

import scripts.train_vq2_lc076_phase6_student_decoder_endpoint as lc076


def test_equal_trajectory_weights() -> None:
    agents = np.array([2, 2, 4, 7, 7, 7], dtype=np.int64)
    weights = lc076.equal_trajectory_weights(
        agents, np.array([2, 4, 7], dtype=np.int64)
    )
    assert np.isclose(weights[agents == 2].sum(), 1 / 3)
    assert np.isclose(weights[agents == 4].sum(), 1 / 3)
    assert np.isclose(weights[agents == 7].sum(), 1 / 3)
    assert np.isclose(weights.sum(), 1.0)


def test_query_agent_split_is_disjoint_and_complete() -> None:
    query = np.arange(18, dtype=np.int64)
    train, validation = lc076.agent_split(query)
    assert len(train) == 14
    assert len(validation) == 4
    assert not set(train) & set(validation)
    assert sorted(np.concatenate((train, validation)).tolist()) == query.tolist()

