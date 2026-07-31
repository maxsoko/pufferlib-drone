from __future__ import annotations

import numpy as np
import torch

import scripts.train_vq2_lc069_phase2_failure_teacher_endpoint as lc069


def test_trajectory_class_weights_balance_agents_and_classes() -> None:
    agents = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    outcome = np.array([1, 0, 0], dtype=np.int8)
    weights = lc069.trajectory_class_weights(
        agents, outcome, np.array([0, 1, 2], dtype=np.int64)
    )
    assert np.isclose(weights[agents == 0].sum(), 0.5)
    assert np.isclose(weights[agents == 1].sum(), 0.25)
    assert np.isclose(weights[agents == 2].sum(), 0.25)
    assert np.isclose(weights.sum(), 1.0)


def test_weighted_ridge_recovers_affine_endpoint() -> None:
    features = torch.tensor(
        [[-2.0, 1.0], [-1.0, 0.5], [0.0, 1.0], [1.0, 2.0], [2.0, -1.0]],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [[1.5, -0.2], [-0.5, 0.7], [0.25, -1.0]], dtype=torch.float64
    )
    target = torch.cat(
        (features, torch.ones(features.shape[0], 1, dtype=torch.float64)), dim=1
    ) @ expected
    solution = lc069.weighted_ridge(
        features, target, torch.ones(features.shape[0], dtype=torch.float64), 0.0
    )
    assert torch.allclose(solution, expected, atol=1e-9, rtol=0.0)


def test_failure_target_and_success_anchor_are_asymmetric() -> None:
    base = torch.tensor([[0.1, -0.2], [0.1, -0.2]])
    parent = torch.tensor([[0.3, 0.4], [0.3, 0.4]])
    teacher = torch.tensor([[0.8, -0.7], [0.8, -0.7]])
    failure = torch.tensor([False, True])
    target = lc069.asymmetric_residual_target(base, parent, teacher, failure)
    assert torch.equal(target[0], parent[0])
    assert torch.allclose(
        target[1], torch.atanh(teacher[1]) - base[1], atol=1e-7, rtol=0.0
    )

