from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import train_vq2_c011_aggregate_warmed_dagger as fit


def test_c011_loader_preserves_intact_histories_and_hashes():
    dataset = fit.C011PublicPhaseDataset()
    assert dataset.agents == 128
    assert dataset.time_steps == 283
    assert dataset.lengths.min() == 276
    assert dataset.lengths.max() == 283


def test_warmed_c011_prepends_exact_prefix():
    suffix = fit.C011PublicPhaseDataset(verify_hashes=False)
    dataset = fit.WarmedDataset(suffix)
    assert dataset.prefix_steps == 208
    assert dataset.time_steps == 491
    observation, action, valid, gate2 = dataset.chunk(
        np.array([0, 64]), 206, 211, device=torch.device("cpu")
    )
    assert observation[:, :2] == pytest.approx(
        torch.from_numpy(dataset.prefix_observation[-2:]).unsqueeze(0).repeat(2, 1, 1)
    )
    assert action[:, :2] == pytest.approx(
        torch.from_numpy(dataset.prefix_action[-2:]).unsqueeze(0).repeat(2, 1, 1)
    )
    assert valid.all()
    assert not gate2[:, :2].any()


def test_aggregate_routes_seven_balanced_groups():
    dataset = fit.AggregateWarmedDataset(verify_hashes=False)
    assert dataset.agents == 448
    assert np.array_equal(np.bincount(dataset.source_for_agent), np.full(7, 64))
    assert np.array_equal(
        np.bincount(dataset.source_for_agent[-56:]), np.full(7, 8)
    )
    observation, action, valid, gate2 = dataset.chunk(
        np.arange(7), 0, 1, device=torch.device("cpu")
    )
    assert observation.shape == (7, 1, 4119)
    assert action.shape == (7, 1, 4)
    assert valid.all()
    assert not gate2.any()
    _, _, suffix_valid, suffix_gate2 = dataset.chunk(
        np.arange(3, 7), 208, 209, device=torch.device("cpu")
    )
    assert suffix_valid.all()
    assert suffix_gate2.all()


def test_fit_contract_is_single_fixed_aggregate_child():
    assert fit.CONFIG.seed == 43012
    assert fit.CONFIG.epochs == 16
    assert fit.CONFIG.validation_agents == 56
    assert fit.CONFIG.learning_rate == pytest.approx(2e-5)
    assert fit.GROUP_NAMES[-2:] == ("c011_true", "c011_alias")
