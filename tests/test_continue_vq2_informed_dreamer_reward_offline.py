from __future__ import annotations

import torch

from pufferlib.vq2_dreamer import RSSMState, VQ2InformedDreamer
from scripts.continue_vq2_informed_dreamer_reward_offline import (
    _prior_features,
    _reward_loss,
)


def test_prior_features_use_deterministic_categorical_latent() -> None:
    model = VQ2InformedDreamer(
        deterministic_size=8,
        stochastic_groups=2,
        stochastic_classes=3,
        distributional_reward=True,
        action_conditioned_reward=True,
    )
    deterministic = torch.randn(2, 4, 8)
    posterior = RSSMState(
        deterministic,
        torch.zeros(2, 4, 2, 3),
        torch.zeros(2, 4, 2, 3),
    )
    prior_logits = torch.randn(2, 4, 2, 3)
    features = _prior_features(model, posterior, prior_logits)
    expected = torch.nn.functional.one_hot(
        prior_logits.argmax(-1), 3
    ).float().flatten(-2)
    assert features.shape == (2, 4, 14)
    assert torch.equal(features[..., :8], deterministic)
    assert torch.equal(features[..., 8:], expected)


def test_reward_target_scale_preserves_order_and_changes_loss() -> None:
    model = VQ2InformedDreamer(
        deterministic_size=8,
        stochastic_groups=2,
        stochastic_classes=3,
        distributional_reward=True,
        action_conditioned_reward=True,
    )
    with torch.no_grad():
        model.reward_predictor[-1].bias.zero_()
        model.reward_predictor[-1].bias[128] = 5.0
    features = torch.zeros(1, 14)
    actions = torch.zeros(1, 4)
    rewards = torch.tensor([0.001])
    unscaled = _reward_loss(model, features, actions, rewards)
    scaled = _reward_loss(
        model, features, actions, rewards, target_scale=64.0
    )
    assert torch.isfinite(unscaled)
    assert torch.isfinite(scaled)
    assert scaled < unscaled
