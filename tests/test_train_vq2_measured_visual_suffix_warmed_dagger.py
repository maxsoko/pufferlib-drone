from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import train_vq2_measured_visual_suffix_warmed_dagger as warmed


@pytest.fixture(scope="module")
def dataset():
    return warmed.WarmedC006PublicPhaseDataset()


def test_warmed_dataset_prepends_exact_legal_prefix(dataset):
    assert dataset.prefix_steps == 208
    assert dataset.time_steps == dataset.suffix.time_steps + 208
    assert np.array_equal(dataset.lengths, dataset.suffix.lengths + 208)
    observation, action, valid, gate2 = dataset.chunk(
        np.array([0, 64]), 0, 8, device=torch.device("cpu")
    )
    expected_observation = torch.from_numpy(dataset.prefix_observation[:8])
    expected_action = torch.from_numpy(dataset.prefix_action[:8])
    assert observation[0] == pytest.approx(expected_observation)
    assert observation[1] == pytest.approx(expected_observation)
    assert action[0] == pytest.approx(expected_action)
    assert valid.all()
    assert not gate2.any()
    assert torch.all(observation[..., -1] == 0.0)


def test_warmed_dataset_handoff_is_exact(dataset):
    agents = np.array([0, 64])
    start = dataset.prefix_steps - 2
    observation, action, valid, gate2 = dataset.chunk(
        agents, start, start + 5, device=torch.device("cpu")
    )
    suffix = dataset.suffix.chunk(agents, 0, 3, device=torch.device("cpu"))
    assert observation[:, :2] == pytest.approx(
        torch.from_numpy(dataset.prefix_observation[-2:]).unsqueeze(0).repeat(2, 1, 1)
    )
    assert action[:, :2] == pytest.approx(
        torch.from_numpy(dataset.prefix_action[-2:]).unsqueeze(0).repeat(2, 1, 1)
    )
    for combined, expected in zip((observation, action, valid, gate2), suffix, strict=True):
        assert combined[:, 2:] == pytest.approx(expected)


def test_config_keeps_single_bounded_fit_contract():
    assert warmed.CONFIG.seed == 43009
    assert warmed.CONFIG.epochs == 16
    assert warmed.CONFIG.learning_rate == pytest.approx(2e-5)
    assert warmed.CONFIG.validation_agents == 40
    assert warmed.PARENT_CHECKPOINT_SHA256.startswith("393f5de6")


def test_base_configuration_does_not_make_dataset_recursive():
    previous = warmed.base.C006PublicPhaseDataset
    try:
        warmed.configure_base()
        dataset = warmed.WarmedC006PublicPhaseDataset(verify_hashes=False)
        assert dataset.prefix_steps == 208
        assert dataset.suffix.__class__ is warmed.UNWARMED_C006_DATASET
    finally:
        warmed.base.C006PublicPhaseDataset = previous
