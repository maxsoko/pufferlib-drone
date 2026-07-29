from __future__ import annotations

import pytest
import torch

from pufferlib.vq2_dreamer import (
    RSSMState,
    VQ2InformedDreamer,
    symexp_twohot_bins,
    twohot_prediction,
    twohot_target,
)
from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE, PRIVILEGED_SIZE


def _small_model() -> VQ2InformedDreamer:
    torch.manual_seed(11)
    return VQ2InformedDreamer(
        deterministic_size=32,
        stochastic_groups=4,
        stochastic_classes=4,
    )


def test_world_model_shapes_and_loss_are_finite() -> None:
    model = _small_model()
    observation = torch.randn(2, 3, ENV_OBS_SIZE)
    observation[..., :4096] = torch.rand(2, 3, 4096)
    action = torch.tanh(torch.randn(2, 3, 4))
    reward = torch.randn(2, 3)
    continuation = torch.ones(2, 3)
    loss, output = model.world_model_loss(
        observation, action, reward, continuation
    )
    assert output.posterior.deterministic.shape == (2, 3, 32)
    assert output.posterior.stochastic.shape == (2, 3, 4, 4)
    assert output.privileged_prediction.shape == (2, 3, PRIVILEGED_SIZE)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.rssm.encoder.image[0].weight.grad is not None


def test_free_nats_floor_can_suppress_and_restore_prior_gradient() -> None:
    observation = torch.randn(2, 3, ENV_OBS_SIZE)
    observation[..., :4096] = torch.rand(2, 3, 4096)
    action = torch.tanh(torch.randn(2, 3, 4))
    reward = torch.randn(2, 3)
    continuation = torch.ones(2, 3)
    floored = _small_model()
    unfloored = _small_model()
    unfloored.load_state_dict(floored.state_dict())

    floored_loss, _ = floored.world_model_loss(
        observation, action, reward, continuation, free_nats=1e6
    )
    floored_loss.total.backward()
    floored_prior_gradient = sum(
        parameter.grad.abs().sum()
        for parameter in floored.rssm.prior.parameters()
        if parameter.grad is not None
    )
    assert float(floored_loss.dynamics_kl.detach()) == pytest.approx(1e6)
    assert floored_loss.dynamics_kl_raw < floored_loss.dynamics_kl
    assert floored_prior_gradient == pytest.approx(0.0)

    unfloored_loss, _ = unfloored.world_model_loss(
        observation, action, reward, continuation, free_nats=0.0
    )
    unfloored_loss.total.backward()
    unfloored_prior_gradient = sum(
        parameter.grad.abs().sum()
        for parameter in unfloored.rssm.prior.parameters()
        if parameter.grad is not None
    )
    assert float(unfloored_loss.dynamics_kl.detach()) == pytest.approx(
        float(unfloored_loss.dynamics_kl_raw.detach())
    )
    assert unfloored_prior_gradient > 0.0


def test_privilege_changes_decoder_loss_not_posterior_or_policy() -> None:
    model = _small_model().eval()
    legal = torch.randn(2, 3, LEGAL_OBS_SIZE)
    information_a = torch.zeros(2, 3, PRIVILEGED_SIZE)
    information_b = torch.full((2, 3, PRIVILEGED_SIZE), 0.75)
    observation_a = torch.cat((legal, information_a), -1)
    observation_b = torch.cat((legal, information_b), -1)
    action = torch.zeros(2, 3, 4)
    with torch.no_grad():
        output_a = model.observe_sequence(
            observation_a, action, deterministic_latent=True
        )
        output_b = model.observe_sequence(
            observation_b, action, deterministic_latent=True
        )
        policy_a, _ = model.policy_step(legal[:, 0], action[:, 0])
        policy_b, _ = model.policy_step(legal[:, 0], action[:, 0])
    assert torch.equal(output_a.posterior.deterministic, output_b.posterior.deterministic)
    assert torch.equal(output_a.posterior.stochastic, output_b.posterior.stochastic)
    assert torch.equal(policy_a, policy_b)
    assert not torch.equal(
        output_a.privileged_prediction - information_a,
        output_b.privileged_prediction - information_b,
    )


def test_detached_context_state_exactly_warms_world_sequence() -> None:
    model = _small_model().eval()
    observation = torch.randn(2, 7, ENV_OBS_SIZE)
    observation[..., :4096] = torch.rand(2, 7, 4096)
    action = torch.tanh(torch.randn(2, 7, 4))
    with torch.no_grad():
        full = model.observe_sequence(
            observation, action, deterministic_latent=True
        )
        context = model.observe_sequence(
            observation[:, :3], action[:, :3], deterministic_latent=True
        )
        initial = RSSMState(
            context.posterior.deterministic[:, -1].detach(),
            context.posterior.stochastic[:, -1].detach(),
            context.posterior.logits[:, -1].detach(),
        )
        world = model.observe_sequence(
            observation[:, 3:],
            action[:, 3:],
            initial_state=initial,
            deterministic_latent=True,
        )
    assert torch.equal(
        full.posterior.deterministic[:, 3:], world.posterior.deterministic
    )
    assert torch.equal(full.posterior.stochastic[:, 3:], world.posterior.stochastic)
    assert torch.equal(full.posterior.logits[:, 3:], world.posterior.logits)


def test_deployed_policy_rejects_native_privileged_tail() -> None:
    model = _small_model().eval()
    with pytest.raises(ValueError, match="legal observation slice only"):
        model.policy_step(torch.zeros(1, ENV_OBS_SIZE), torch.zeros(1, 4))


def test_untrained_deployed_policy_mean_is_centered_hover() -> None:
    model = _small_model().eval()
    with torch.no_grad():
        action, _ = model.policy_step(
            torch.zeros(2, LEGAL_OBS_SIZE), torch.zeros(2, 4)
        )
    assert torch.equal(action, torch.zeros_like(action))


def test_training_policy_initial_std_matches_local_collection_support() -> None:
    model = _small_model().eval()
    with torch.no_grad():
        distribution, _ = model.policy_distribution_step(
            torch.zeros(2, LEGAL_OBS_SIZE), torch.zeros(2, 4)
        )
    assert torch.allclose(
        distribution.stddev, torch.full_like(distribution.stddev, 0.20)
    )


def test_pinned_bounded_normal_bounds_mean_not_sample() -> None:
    model = VQ2InformedDreamer(
        deterministic_size=32,
        stochastic_groups=4,
        stochastic_classes=4,
        actor_distribution_mode="dreamerv3_bounded_normal",
        actor_initial_std=0.20,
    ).eval()
    features = torch.randn(5, model.rssm.feature_size)
    with torch.no_grad():
        distribution = model.actor_distribution(features)
        action = model.deterministic_actor_action(distribution)
    assert torch.equal(action, distribution.mean)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    assert torch.allclose(
        distribution.stddev, torch.full_like(distribution.stddev, 0.20)
    )


def test_pinned_percentile_advantage_updates_non_debiased_ema() -> None:
    model = VQ2InformedDreamer(
        deterministic_size=32,
        stochastic_groups=4,
        stochastic_classes=4,
        actor_distribution_mode="dreamerv3_bounded_normal",
    )
    start = model.rssm.initial(8, device="cpu")
    loss = model.imagination_loss(
        start,
        horizon=4,
        advantage_normalization="dreamerv3_percentile",
    )
    assert loss.return_scale == pytest.approx(1.0)
    assert model.return_normalizer_high != 0.0
    assert model.return_normalizer_low != 0.0


def test_dreamerv3_twohot_support_is_symmetric_and_interpolates() -> None:
    bins = symexp_twohot_bins()
    assert bins.shape == (255,)
    assert bins[127] == 0.0
    assert torch.equal(bins[:127], -bins[128:].flip(0))
    target = torch.tensor([-0.05, 0.0, 0.05, 30.0])
    encoded = twohot_target(target, bins)
    assert torch.allclose(encoded.sum(-1), torch.ones(4))
    assert torch.allclose((encoded * bins).sum(-1), target, atol=2e-5)


def test_distributional_reward_head_starts_at_exact_zero() -> None:
    model = VQ2InformedDreamer(
        deterministic_size=32,
        stochastic_groups=4,
        stochastic_classes=4,
        distributional_reward=True,
    )
    features = torch.randn(5, model.rssm.feature_size)
    logits = model.reward_predictor(features)
    assert torch.equal(logits, torch.zeros_like(logits))
    assert torch.equal(
        twohot_prediction(logits, model.reward_bins), torch.zeros(5)
    )
    assert torch.equal(model.predict_reward(features), torch.zeros(5))


def test_action_conditioned_reward_head_has_explicit_causal_skip() -> None:
    model = VQ2InformedDreamer(
        deterministic_size=32,
        stochastic_groups=4,
        stochastic_classes=4,
        distributional_reward=True,
        action_conditioned_reward=True,
    )
    features = torch.randn(5, model.rssm.feature_size)
    actions = torch.randn(5, 4)
    assert model.reward_predictor[0].in_features == model.rssm.feature_size + 4
    with pytest.raises(ValueError, match="requires action"):
        model.predict_reward(features)
    assert torch.equal(model.predict_reward(features, actions), torch.zeros(5))


def test_imagination_produces_actor_and_critic_gradients() -> None:
    model = _small_model()
    start = model.rssm.initial(3, device="cpu")
    loss = model.imagination_loss(start, horizon=4)
    assert torch.isfinite(loss.actor)
    assert torch.isfinite(loss.critic)
    assert torch.isfinite(loss.return_scale)
    assert loss.return_scale >= 0
    assert torch.isfinite(loss.mean_continuation_weight)
    assert 0.0 <= loss.mean_continuation_weight <= 1.0
    loss.actor.backward()
    assert next(model.actor.parameters()).grad is not None
    model.zero_grad(set_to_none=True)
    loss.critic.backward()
    assert next(model.critic.parameters()).grad is not None


def test_informed_decoder_progress_is_training_only_and_differentiable_actor() -> None:
    model = _small_model()
    start = model.rssm.initial(3, device="cpu")
    features = start.features
    assert model.decoded_gate_range(features).shape == (3,)
    assert torch.equal(
        model.decoded_progress_reward(features, features), torch.zeros(3)
    )
    loss = model.imagination_loss(
        start,
        horizon=4,
        reward_source="informed_decoder_progress",
    )
    assert torch.isfinite(loss.actor)
    assert torch.isfinite(loss.critic)
    loss.actor.backward()
    assert next(model.actor.parameters()).grad is not None


def test_decoded_skydreamer_reward_matches_paper_task_terms() -> None:
    model = _small_model()
    model.privileged_decoder = torch.nn.Linear(model.rssm.feature_size, 34)
    decoder_head = model.privileged_decoder
    decoder_head.weight.data.zero_()
    decoder_head.bias.data.zero_()

    current = torch.zeros(1, model.rssm.feature_size)
    following = torch.ones(1, model.rssm.feature_size)
    # Use the final linear weights to produce two exact normalized decoder
    # targets: phase advances one of six gates, y/z are 0.25/-0.50 m, radius
    # is 1 m, and the next body-rate L1 norm is 1 rad/s.
    decoder_head.bias.data[3:6] = torch.tanh(
        torch.tensor([0.1, 0.025, -0.05])
    )
    decoder_head.bias.data[33] = 1.0 / 3.0
    decoder_head.weight.data[16, 0] = 0.05
    decoder_head.weight.data[32, 0] = 1.0 / 6.0
    terms = model.decoded_skydreamer_reward_terms(
        current, following, control_frequency_hz=64.0
    )
    current_range = torch.tensor(1.0**2 + 0.25**2 + 0.5**2).sqrt()
    next_range = current_range
    expected_progress = 5.0 * (current_range - next_range)
    expected_rate = torch.expm1(torch.tensor(1.0)) / (2.0 * 64.0 * 1.0e5)
    expected_gate = 30.0 * (1.0 - 0.5 / 1.0)
    assert terms.total.item() == pytest.approx(
        (expected_progress - expected_rate + expected_gate).item(), rel=1e-5
    )
    assert terms.progress.item() == pytest.approx(expected_progress.item())
    assert terms.rate_penalty.item() == pytest.approx(expected_rate.item())
    assert terms.gate.item() == pytest.approx(expected_gate)
    assert terms.crossed.item() is True
    assert torch.equal(
        model.decoded_skydreamer_reward(current, following), terms.total
    )

    with pytest.raises(ValueError, match="control frequency"):
        model.decoded_skydreamer_reward(current, following, control_frequency_hz=0)


def test_action_effort_is_training_only_and_validated() -> None:
    model = _small_model()
    start = model.rssm.initial(3, device="cpu")
    torch.manual_seed(7)
    loss = model.imagination_loss(
        start,
        horizon=4,
        reward_source="informed_decoder_progress",
        action_effort_weights=(0.0, 0.02, 0.05, 0.4),
    )
    assert torch.isfinite(loss.mean_effort)
    assert loss.mean_effort > 0.0
    with pytest.raises(ValueError, match="match action size"):
        model.imagination_loss(start, horizon=4, action_effort_weights=(0.0, 0.0))
    with pytest.raises(ValueError, match="cannot be negative"):
        model.imagination_loss(
            start,
            horizon=4,
            action_effort_weights=(0.0, 0.0, -0.1, 0.0),
        )
