from __future__ import annotations

import numpy as np
import torch

from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseResidualActor
from scripts.train_vq2_vg064_indexed_intervention_ridge import (
    ACTION_SIZE,
    FEATURE_DTYPE,
    HIDDEN_SIZE,
    TARGET_PHASES,
    ridge_solution,
    target_pre_tanh_residual,
    validation_mask,
)


def test_validation_split_is_agent_stable() -> None:
    agents = np.arange(512, dtype=np.uint16)
    selected = validation_mask(agents)
    assert selected.sum() == 64
    expected = np.concatenate([np.arange(start, start + 8) for start in range(0, 512, 64)])
    assert np.array_equal(np.flatnonzero(selected), expected)
    assert set((agents[selected] % 8).tolist()) == set(range(8))


def test_pre_tanh_target_round_trips_unsaturated_teacher() -> None:
    base = torch.tensor([[0.2, -0.3, 0.1, 0.0]])
    teacher = torch.tensor([[0.5, -0.4, 0.2, 0.0]])
    residual = target_pre_tanh_residual(base, teacher)
    torch.testing.assert_close(torch.tanh(base + residual), teacher)


def test_ridge_solution_recovers_linear_map() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1024, HIDDEN_SIZE, generator=generator, dtype=torch.float64)
    truth = torch.randn(HIDDEN_SIZE, ACTION_SIZE, generator=generator, dtype=torch.float64) * 0.01
    y = x @ truth
    solved = ridge_solution(x.T @ x, x.T @ y, 1e-8)
    torch.testing.assert_close(solved, truth, atol=1e-8, rtol=1e-6)


def test_indexed_actor_has_one_independent_head_per_target_phase() -> None:
    actor = VQ2IndexedPhaseResidualActor(hidden_size=HIDDEN_SIZE)
    assert actor.indexed_phase_action_residual.shape == (17, ACTION_SIZE, HIDDEN_SIZE)
    assert TARGET_PHASES == tuple(range(1, 12))


def test_structured_feature_fields_require_contiguous_copies() -> None:
    records = np.zeros(3, dtype=FEATURE_DTYPE)
    assert records["base_pre_tanh"].strides[0] == FEATURE_DTYPE.itemsize
    contiguous = np.array(records["base_pre_tanh"], dtype=np.float32, copy=True)
    assert contiguous.flags.c_contiguous
    tensor = torch.from_numpy(contiguous)
    assert tensor.shape == (3, ACTION_SIZE)
