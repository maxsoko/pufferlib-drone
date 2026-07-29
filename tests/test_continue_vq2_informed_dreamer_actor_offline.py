from __future__ import annotations

import pytest
import torch

from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer
from scripts.continue_vq2_informed_dreamer_actor_offline import (
    _configure_actor_training,
    _repeat_imagination_starts,
    _select_imagination_starts,
)


def _posterior() -> RSSMState:
    return RSSMState(
        torch.arange(2 * 4 * 3).reshape(2, 4, 3).float(),
        torch.arange(2 * 4 * 2 * 2).reshape(2, 4, 2, 2).float(),
        torch.arange(2 * 4 * 2 * 2).reshape(2, 4, 2, 2).float(),
    )


def test_last_imagination_starts_preserve_legacy_batch() -> None:
    selected = _select_imagination_starts(_posterior(), mode="last", count=0)
    assert selected.deterministic.shape == (2, 3)
    assert torch.equal(selected.deterministic, _posterior().deterministic[:, -1])


def test_all_imagination_starts_flatten_time_with_hardware_bound() -> None:
    torch.manual_seed(3)
    selected = _select_imagination_starts(_posterior(), mode="all", count=5)
    assert selected.deterministic.shape == (5, 3)
    assert selected.stochastic.shape == (5, 2, 2)
    assert selected.logits.shape == (5, 2, 2)


def test_imagination_start_selection_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="does not accept"):
        _select_imagination_starts(_posterior(), mode="last", count=1)
    with pytest.raises(ValueError, match="must lie"):
        _select_imagination_starts(_posterior(), mode="all", count=9)


def test_imagination_start_repeats_preserve_each_latent() -> None:
    selected = _select_imagination_starts(_posterior(), mode="last", count=0)
    repeated = _repeat_imagination_starts(selected, 3)
    assert repeated.deterministic.shape == (6, 3)
    assert torch.equal(repeated.deterministic[0], selected.deterministic[0])
    assert torch.equal(repeated.deterministic[2], selected.deterministic[0])
    assert torch.equal(repeated.deterministic[3], selected.deterministic[1])
    with pytest.raises(ValueError, match="must be positive"):
        _repeat_imagination_starts(selected, 0)


def test_restricted_actor_training_masks_all_but_selected_mean_rows() -> None:
    model = VQ2InformedDreamer(
        deterministic_size=16,
        stochastic_groups=2,
        stochastic_classes=2,
    )
    parameters = _configure_actor_training(model, (0, 2), weight_decay=0.0)
    assert parameters == [model.actor[-1].weight, model.actor[-1].bias]
    distribution = model.actor_distribution(
        torch.randn(8, model.rssm.feature_size)
    )
    distribution.mean.sum().backward()
    weight_gradient = model.actor[-1].weight.grad
    bias_gradient = model.actor[-1].bias.grad
    assert weight_gradient is not None and bias_gradient is not None
    assert torch.count_nonzero(weight_gradient[0]) > 0
    assert torch.count_nonzero(weight_gradient[2]) > 0
    assert torch.count_nonzero(weight_gradient[1]) == 0
    assert torch.count_nonzero(weight_gradient[3:]) == 0
    assert torch.equal(
        bias_gradient,
        torch.tensor([8.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
