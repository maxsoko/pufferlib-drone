"""Compact Informed-Dreamer components for the VQ2 Puffer environment.

This is deliberately independent from Puffer PPO. Puffer supplies native
rollouts; the model learns an RSSM from replay and the actor learns from latent
imagination. Privileged information is used only as a decoder target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from pufferlib.vq2_informed import (
    LEGAL_OBS_SIZE,
    PRIVILEGED_SIZE,
    VQ2VisualEncoder,
    split_environment_observation,
)


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.expm1(torch.abs(value))


def symexp_twohot_bins(
    count: int = 255,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """DreamerV3 scalar support pinned by the SkyDreamer implementation."""

    if count < 3 or count % 2 != 1:
        raise ValueError("symexp two-hot support requires an odd count >= 3")
    half = torch.linspace(-20.0, 0.0, (count - 1) // 2 + 1, device=device)
    half = symexp(half)
    return torch.cat((half, -half[:-1].flip(0)))


def twohot_target(target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate a scalar onto adjacent ordered support bins."""

    target = target.float().clamp(float(bins[0]), float(bins[-1]))
    above = torch.searchsorted(bins, target, right=True).clamp(
        0, bins.numel() - 1
    )
    below = (above - 1).clamp(0, bins.numel() - 1)
    below_value = bins[below]
    above_value = bins[above]
    equal = below == above
    distance_below = torch.where(
        equal, torch.ones_like(target), (below_value - target).abs()
    )
    distance_above = torch.where(
        equal, torch.ones_like(target), (above_value - target).abs()
    )
    total = distance_below + distance_above
    weight_below = distance_above / total
    weight_above = distance_below / total
    encoded = torch.zeros(*target.shape, bins.numel(), device=target.device)
    encoded.scatter_add_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    encoded.scatter_add_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    return encoded


def twohot_prediction(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Symmetric weighted sum, preserving exact zero for uniform logits."""

    probabilities = logits.float().softmax(-1)
    midpoint = (bins.numel() - 1) // 2
    negative = probabilities[..., :midpoint] * bins[:midpoint]
    center = probabilities[..., midpoint] * bins[midpoint]
    positive = probabilities[..., midpoint + 1 :] * bins[midpoint + 1 :]
    return center + (negative.flip(-1) + positive).sum(-1)


def twohot_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor
) -> torch.Tensor:
    encoded = twohot_target(target, bins)
    return -(encoded * F.log_softmax(logits.float(), -1)).sum(-1).mean()


class RSSMState(NamedTuple):
    deterministic: torch.Tensor
    stochastic: torch.Tensor
    logits: torch.Tensor

    @property
    def features(self) -> torch.Tensor:
        return torch.cat((self.deterministic, self.stochastic.flatten(-2)), -1)

    def detached(self) -> "RSSMState":
        return RSSMState(*(value.detach() for value in self))


@dataclass(frozen=True)
class WorldModelOutput:
    posterior: RSSMState
    prior_logits: torch.Tensor
    privileged_prediction: torch.Tensor
    reward_prediction: torch.Tensor
    reward_logits: torch.Tensor
    continue_logit: torch.Tensor


@dataclass(frozen=True)
class WorldModelLoss:
    total: torch.Tensor
    privileged: torch.Tensor
    reward: torch.Tensor
    continuation: torch.Tensor
    dynamics_kl: torch.Tensor
    representation_kl: torch.Tensor
    dynamics_kl_raw: torch.Tensor
    representation_kl_raw: torch.Tensor


@dataclass(frozen=True)
class ImaginationLoss:
    actor: torch.Tensor
    critic: torch.Tensor
    reinforce: torch.Tensor
    entropy: torch.Tensor
    smoothness: torch.Tensor
    mean_effort: torch.Tensor
    mean_return: torch.Tensor
    return_scale: torch.Tensor
    mean_continuation_weight: torch.Tensor


@dataclass(frozen=True)
class DecodedSkyDreamerReward:
    total: torch.Tensor
    progress: torch.Tensor
    rate_penalty: torch.Tensor
    gate: torch.Tensor
    crossed: torch.Tensor


def _mlp(input_size: int, output_size: int, hidden_size: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, output_size),
    )


class CategoricalRSSM(nn.Module):
    """Dreamer-style recurrent state-space model with discrete latents."""

    def __init__(
        self,
        *,
        action_size: int = 4,
        encoder_size: int = 256,
        deterministic_size: int = 256,
        stochastic_groups: int = 16,
        stochastic_classes: int = 16,
    ) -> None:
        super().__init__()
        self.action_size = action_size
        self.deterministic_size = deterministic_size
        self.stochastic_groups = stochastic_groups
        self.stochastic_classes = stochastic_classes
        self.stochastic_size = stochastic_groups * stochastic_classes
        self.feature_size = deterministic_size + self.stochastic_size
        self.encoder = VQ2VisualEncoder(encoder_size)
        self.sequence = nn.GRUCell(
            self.stochastic_size + action_size, deterministic_size
        )
        self.prior = _mlp(
            deterministic_size,
            self.stochastic_size,
            hidden_size=deterministic_size,
        )
        self.posterior = _mlp(
            deterministic_size + encoder_size,
            self.stochastic_size,
            hidden_size=deterministic_size,
        )

    def initial(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> RSSMState:
        deterministic = torch.zeros(
            batch_size, self.deterministic_size, device=device, dtype=dtype
        )
        logits = torch.zeros(
            batch_size,
            self.stochastic_groups,
            self.stochastic_classes,
            device=device,
            dtype=dtype,
        )
        stochastic = F.one_hot(
            torch.zeros(
                batch_size,
                self.stochastic_groups,
                device=device,
                dtype=torch.long,
            ),
            self.stochastic_classes,
        ).to(dtype)
        return RSSMState(deterministic, stochastic, logits)

    def _reshape_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.reshape(
            *logits.shape[:-1], self.stochastic_groups, self.stochastic_classes
        )

    def _latent(self, logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
        if deterministic:
            index = logits.argmax(-1)
            return F.one_hot(index, self.stochastic_classes).to(logits.dtype)
        return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)

    def observe_step(
        self,
        previous: RSSMState,
        previous_action: torch.Tensor,
        legal_observation: torch.Tensor,
        *,
        deterministic_latent: bool = False,
    ) -> tuple[RSSMState, torch.Tensor]:
        if legal_observation.shape[-1] != LEGAL_OBS_SIZE:
            raise ValueError("RSSM posterior accepts competition-legal observations only")
        sequence_input = torch.cat(
            (previous.stochastic.flatten(-2), previous_action), -1
        )
        deterministic = self.sequence(sequence_input, previous.deterministic)
        prior_logits = self._reshape_logits(self.prior(deterministic))
        encoded = self.encoder(legal_observation)
        posterior_logits = self._reshape_logits(
            self.posterior(torch.cat((deterministic, encoded), -1))
        )
        stochastic = self._latent(posterior_logits, deterministic_latent)
        return RSSMState(deterministic, stochastic, posterior_logits), prior_logits

    def imagine_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        *,
        deterministic_latent: bool = False,
    ) -> RSSMState:
        sequence_input = torch.cat((previous.stochastic.flatten(-2), action), -1)
        deterministic = self.sequence(sequence_input, previous.deterministic)
        logits = self._reshape_logits(self.prior(deterministic))
        stochastic = self._latent(logits, deterministic_latent)
        return RSSMState(deterministic, stochastic, logits)


class VQ2InformedDreamer(nn.Module):
    """World model, latent actor, and critic for VQ2."""

    def __init__(
        self,
        *,
        action_size: int = 4,
        deterministic_size: int = 256,
        stochastic_groups: int = 16,
        stochastic_classes: int = 16,
        actor_initial_std: float = 0.20,
        actor_distribution_mode: str = "legacy_tanh_normal",
        distributional_reward: bool = False,
        action_conditioned_reward: bool = False,
        reward_bins: int = 255,
    ) -> None:
        super().__init__()
        self.action_size = action_size
        if actor_distribution_mode not in (
            "legacy_tanh_normal",
            "dreamerv3_bounded_normal",
        ):
            raise ValueError(
                f"unsupported actor distribution mode: {actor_distribution_mode}"
            )
        self.actor_distribution_mode = actor_distribution_mode
        self.rssm = CategoricalRSSM(
            action_size=action_size,
            deterministic_size=deterministic_size,
            stochastic_groups=stochastic_groups,
            stochastic_classes=stochastic_classes,
        )
        feature_size = self.rssm.feature_size
        self.distributional_reward = distributional_reward
        self.action_conditioned_reward = action_conditioned_reward
        self.privileged_decoder = _mlp(feature_size, PRIVILEGED_SIZE)
        self.reward_predictor = _mlp(
            feature_size + (action_size if action_conditioned_reward else 0),
            reward_bins if distributional_reward else 1,
        )
        if distributional_reward:
            self.register_buffer(
                "reward_bins", symexp_twohot_bins(reward_bins), persistent=True
            )
            # The pinned DreamerV3 reward head uses output scale zero.
            nn.init.zeros_(self.reward_predictor[-1].weight)
            nn.init.zeros_(self.reward_predictor[-1].bias)
        else:
            self.reward_bins = None
        self.continue_predictor = _mlp(feature_size, 1)
        self.actor = _mlp(feature_size, 2 * action_size)
        # The normalized CTBR convention is centered on hover and level
        # attitude. A zeroed policy head therefore begins with a stable mean
        # while its learned distribution and collector noise retain exploration.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        if actor_distribution_mode == "legacy_tanh_normal":
            if actor_initial_std <= 0.05:
                raise ValueError("actor_initial_std must exceed the 0.05 std floor")
            raw_std_bias = torch.log(
                torch.expm1(torch.tensor(actor_initial_std - 0.05))
            ) - 0.5
        else:
            if not 0.1 < actor_initial_std < 1.0:
                raise ValueError(
                    "bounded-normal actor_initial_std must lie in (0.1,1.0)"
                )
            probability = (actor_initial_std - 0.1) / 0.9
            raw_std_bias = torch.logit(torch.tensor(probability)) - 2.0
        with torch.no_grad():
            self.actor[-1].bias[action_size:].fill_(float(raw_std_bias))
        self.critic = _mlp(feature_size, 1)
        self.target_critic = _mlp(feature_size, 1)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        # SkyDreamer's pinned DreamerV3 configuration uses a non-debiased
        # 5th/95th percentile EMA for return scaling. These are deliberately
        # non-persistent model buffers so historical state_dicts remain exact;
        # training checkpoints store them as explicit optimizer-side state.
        self.register_buffer("return_normalizer_low", torch.tensor(0.0), persistent=False)
        self.register_buffer("return_normalizer_high", torch.tensor(0.0), persistent=False)
        # Mean-normalized informed reconstruction weights. Gate-relative pose,
        # camera extrinsics, and recurrent course phase are the targets that
        # force the posterior to explain visual variation instead of predicting
        # the near-zero mean of easier rate/RPM channels.
        privileged_weights = torch.tensor(
            [
                1, 1, 1,          # world position
                8, 8, 8,          # active-gate-relative body position
                1, 1, 1,          # world velocity
                1, 1, 1,          # body velocity
                2, 2, 2, 2,       # attitude quaternion
                1, 1, 1,          # body rates
                1, 1, 1, 1,       # motor RPM
                8, 8, 8,          # camera extrinsics
                2, 2, 2, 2, 2, 2, # dynamics
                8,                 # recurrent ordered phase
                2,                 # active aperture
            ],
            dtype=torch.float32,
        )
        privileged_weights /= privileged_weights.mean()
        self.register_buffer(
            "privileged_weights", privileged_weights, persistent=False
        )

    def reward_logits(
        self, features: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.action_conditioned_reward:
            if action is None:
                raise ValueError("action-conditioned reward head requires action")
            if action.shape[:-1] != features.shape[:-1]:
                raise ValueError("reward action and latent features are misaligned")
            features = torch.cat((features, action), -1)
        return self.reward_predictor(features)

    def predict_reward(
        self, features: torch.Tensor, action: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = self.reward_logits(features, action)
        if self.distributional_reward:
            assert self.reward_bins is not None
            return twohot_prediction(logits, self.reward_bins)
        return symexp(logits.squeeze(-1))

    def decoded_gate_range(self, features: torch.Tensor) -> torch.Tensor:
        """Decode the training-only active-gate range in metres.

        The decoder is densely supervised from native state during training,
        but its output never enters the deployed actor or RSSM. Clamping only
        guards the inverse of the native tanh normalization.
        """

        privileged = self.privileged_decoder(features)
        gate_body = privileged[..., 3:6].clamp(-0.999999, 0.999999)
        return (10.0 * torch.atanh(gate_body)).square().sum(-1).sqrt()

    def decoded_progress_reward(
        self,
        current_features: torch.Tensor,
        next_features: torch.Tensor,
        *,
        progress_scale: float = 5.0,
    ) -> torch.Tensor:
        """Apply SkyDreamer's known progress term to decoded latent state."""

        if progress_scale <= 0.0:
            raise ValueError("decoded progress scale must be positive")
        return progress_scale * (
            self.decoded_gate_range(current_features)
            - self.decoded_gate_range(next_features)
        )

    def decoded_skydreamer_reward_terms(
        self,
        current_features: torch.Tensor,
        next_features: torch.Tensor,
        *,
        control_frequency_hz: float = 64.0,
    ) -> DecodedSkyDreamerReward:
        """Decode each SkyDreamer task-reward term for offline diagnostics.

        The native privileged-target ABI normalizes active-gate body position
        by ``tanh(metres / 10)``, body rates by ``20 rad/s``, ordered course
        phase by ``gate_index / 6``, and aperture radius by ``3 m``. A decoded
        half-gate-or-larger phase increment is the gate-pass event. This is
        necessary because a true pass immediately changes the supervised
        active-gate target to the next gate; old-gate x therefore never becomes
        negative in replay. The centered bonus scores the pre-transition gate
        offset with the paper's Chebyshev distance. None of these decoder
        outputs enters the actor, RSSM, or deployed callable.
        """

        if control_frequency_hz <= 0.0:
            raise ValueError("control frequency must be positive")
        current_privileged = self.privileged_decoder(current_features)
        next_privileged = self.privileged_decoder(next_features)
        current_gate = 10.0 * torch.atanh(
            current_privileged[..., 3:6].clamp(-0.999999, 0.999999)
        )
        next_gate = 10.0 * torch.atanh(
            next_privileged[..., 3:6].clamp(-0.999999, 0.999999)
        )

        current_range = current_gate.square().sum(-1).sqrt()
        next_range = next_gate.square().sum(-1).sqrt()
        progress_reward = 5.0 * (current_range - next_range)

        next_body_rate = 20.0 * next_privileged[..., 16:19].clamp(-1.0, 1.0)
        rate_l1 = next_body_rate.abs().sum(-1).clamp_max(17.0)
        rate_penalty = torch.expm1(rate_l1) / (
            2.0 * float(control_frequency_hz) * 1.0e5
        )

        current_phase = current_privileged[..., 32].clamp(0.0, 1.0)
        next_phase = next_privileged[..., 32].clamp(0.0, 1.0)
        crossed = (next_phase - current_phase) >= (0.5 / 6.0)
        crossing_offset = current_gate[..., 1:3].abs().amax(-1)
        aperture = (3.0 * current_privileged[..., 33]).clamp(1.0e-4, 3.0)
        centered_crossing = (1.0 - crossing_offset / aperture).clamp(0.0, 1.0)
        gate_reward = 30.0 * torch.where(
            crossed, centered_crossing, torch.zeros_like(centered_crossing)
        )
        return DecodedSkyDreamerReward(
            total=progress_reward - rate_penalty + gate_reward,
            progress=progress_reward,
            rate_penalty=rate_penalty,
            gate=gate_reward,
            crossed=crossed,
        )

    def decoded_skydreamer_reward(
        self,
        current_features: torch.Tensor,
        next_features: torch.Tensor,
        *,
        control_frequency_hz: float = 64.0,
    ) -> torch.Tensor:
        """Evaluate SkyDreamer's task reward from training-only predictions."""

        return self.decoded_skydreamer_reward_terms(
            current_features,
            next_features,
            control_frequency_hz=control_frequency_hz,
        ).total

    def observe_sequence(
        self,
        environment_observation: torch.Tensor,
        previous_action: torch.Tensor,
        *,
        initial_state: RSSMState | None = None,
        deterministic_latent: bool = False,
    ) -> WorldModelOutput:
        if environment_observation.ndim != 3:
            raise ValueError("world-model batches must have shape [batch, time, values]")
        if previous_action.shape[:2] != environment_observation.shape[:2]:
            raise ValueError("previous actions must align with observation sequences")
        separated = split_environment_observation(environment_observation)
        batch_size, sequence_length = separated.legal.shape[:2]
        if initial_state is None:
            state = self.rssm.initial(
                batch_size,
                device=environment_observation.device,
                dtype=environment_observation.dtype,
            )
        else:
            if initial_state.deterministic.shape[0] != batch_size:
                raise ValueError("initial RSSM state batch size does not match sequence")
            state = initial_state
        states: list[RSSMState] = []
        priors: list[torch.Tensor] = []
        for step in range(sequence_length):
            state, prior = self.rssm.observe_step(
                state,
                previous_action[:, step],
                separated.legal[:, step],
                deterministic_latent=deterministic_latent,
            )
            states.append(state)
            priors.append(prior)
        posterior = RSSMState(
            *(torch.stack([getattr(state, field) for state in states], 1)
              for field in RSSMState._fields)
        )
        prior_logits = torch.stack(priors, 1)
        features = posterior.features
        reward_logits = self.reward_logits(features, previous_action)
        if self.distributional_reward:
            assert self.reward_bins is not None
            reward_prediction = twohot_prediction(reward_logits, self.reward_bins)
        else:
            reward_prediction = reward_logits.squeeze(-1)
        return WorldModelOutput(
            posterior=posterior,
            prior_logits=prior_logits,
            privileged_prediction=self.privileged_decoder(features),
            reward_prediction=reward_prediction,
            reward_logits=reward_logits,
            continue_logit=self.continue_predictor(features).squeeze(-1),
        )

    @staticmethod
    def _categorical_kl(
        posterior_logits: torch.Tensor, prior_logits: torch.Tensor
    ) -> torch.Tensor:
        posterior_log_prob = F.log_softmax(posterior_logits, -1)
        prior_log_prob = F.log_softmax(prior_logits, -1)
        posterior_prob = posterior_log_prob.exp()
        return (posterior_prob * (posterior_log_prob - prior_log_prob)).sum(-1).sum(-1)

    def world_model_loss(
        self,
        environment_observation: torch.Tensor,
        previous_action: torch.Tensor,
        reward: torch.Tensor,
        continuation: torch.Tensor,
        *,
        initial_state: RSSMState | None = None,
        free_nats: float = 1.0,
        dynamics_weight: float = 1.0,
        representation_weight: float = 0.1,
        deterministic_latent: bool = False,
    ) -> tuple[WorldModelLoss, WorldModelOutput]:
        output = self.observe_sequence(
            environment_observation,
            previous_action,
            initial_state=initial_state,
            deterministic_latent=deterministic_latent,
        )
        information = split_environment_observation(
            environment_observation
        ).privileged
        privileged_error = F.smooth_l1_loss(
            output.privileged_prediction, information, reduction="none"
        )
        privileged_loss = (
            privileged_error * self.privileged_weights.to(privileged_error.dtype)
        ).mean()
        if self.distributional_reward:
            assert self.reward_bins is not None
            reward_loss = twohot_cross_entropy(
                output.reward_logits, reward, self.reward_bins
            )
        else:
            reward_loss = F.smooth_l1_loss(
                output.reward_prediction, symlog(reward)
            )
        continuation_loss = F.binary_cross_entropy_with_logits(
            output.continue_logit, continuation
        )
        dynamics_kl_values = self._categorical_kl(
            output.posterior.logits.detach(), output.prior_logits
        )
        representation_kl_values = self._categorical_kl(
            output.posterior.logits, output.prior_logits.detach()
        )
        dynamics_kl = dynamics_kl_values.clamp_min(free_nats).mean()
        representation_kl = representation_kl_values.clamp_min(free_nats).mean()
        total = (
            privileged_loss
            + reward_loss
            + continuation_loss
            + dynamics_weight * dynamics_kl
            + representation_weight * representation_kl
        )
        return (
            WorldModelLoss(
                total=total,
                privileged=privileged_loss,
                reward=reward_loss,
                continuation=continuation_loss,
                dynamics_kl=dynamics_kl,
                representation_kl=representation_kl,
                dynamics_kl_raw=dynamics_kl_values.mean(),
                representation_kl_raw=representation_kl_values.mean(),
            ),
            output,
        )

    def actor_distribution(self, features: torch.Tensor) -> Normal:
        raw_mean, raw_std = self.actor(features).chunk(2, -1)
        if self.actor_distribution_mode == "dreamerv3_bounded_normal":
            # Pinned DreamerV3 `bounded_normal`: bound the distribution mean,
            # not the sampled action, and constrain stddev to [0.1, 1.0].
            mean = torch.tanh(raw_mean)
            std = 0.9 * torch.sigmoid(raw_std + 2.0) + 0.1
        else:
            mean = 5.0 * torch.tanh(raw_mean / 5.0)
            std = F.softplus(raw_std + 0.5) + 0.05
        return Normal(mean, std)

    def deterministic_actor_action(self, distribution: Normal) -> torch.Tensor:
        if self.actor_distribution_mode == "dreamerv3_bounded_normal":
            return distribution.mean
        return torch.tanh(distribution.mean)

    def sample_actor_action(self, distribution: Normal) -> torch.Tensor:
        sample = distribution.sample()
        if self.actor_distribution_mode == "dreamerv3_bounded_normal":
            return sample
        return torch.tanh(sample)

    def policy_step(
        self,
        legal_observation: torch.Tensor,
        previous_action: torch.Tensor,
        previous_state: RSSMState | None = None,
    ) -> tuple[torch.Tensor, RSSMState]:
        """Deterministic deployed posterior update and complete neural action."""

        distribution, state = self.policy_distribution_step(
            legal_observation, previous_action, previous_state
        )
        return self.deterministic_actor_action(distribution), state

    def policy_distribution_step(
        self,
        legal_observation: torch.Tensor,
        previous_action: torch.Tensor,
        previous_state: RSSMState | None = None,
    ) -> tuple[Normal, RSSMState]:
        """Posterior update plus the learned stochastic training policy."""

        if legal_observation.shape[-1] != LEGAL_OBS_SIZE:
            raise ValueError("deployed policy accepts the legal observation slice only")
        if previous_state is None:
            previous_state = self.rssm.initial(
                legal_observation.shape[0],
                device=legal_observation.device,
                dtype=legal_observation.dtype,
            )
        state, _ = self.rssm.observe_step(
            previous_state,
            previous_action,
            legal_observation,
            deterministic_latent=True,
        )
        return self.actor_distribution(state.features), state

    def imagination_loss(
        self,
        starts: RSSMState,
        *,
        horizon: int = 16,
        gamma: float = 0.997,
        lambda_: float = 0.95,
        entropy_weight: float = 1e-5,
        smoothness_weight: float = 0.002,
        reward_source: str = "learned",
        action_effort_weights: tuple[float, float, float, float] | None = None,
        advantage_normalization: str = "legacy_center_std",
        update_return_normalizer: bool = True,
        control_frequency_hz: float = 64.0,
    ) -> ImaginationLoss:
        """REINFORCE actor and value loss over prior-only latent rollouts."""

        if horizon < 2:
            raise ValueError("imagination horizon must be at least two")
        if reward_source not in (
            "learned",
            "informed_decoder_progress",
            "informed_decoder_skydreamer",
        ):
            raise ValueError(f"unsupported imagination reward source: {reward_source}")
        if advantage_normalization not in (
            "legacy_center_std",
            "dreamerv3_percentile",
        ):
            raise ValueError(
                f"unsupported advantage normalization: {advantage_normalization}"
            )
        if action_effort_weights is None:
            effort_weights = None
        else:
            if len(action_effort_weights) != self.action_size:
                raise ValueError("action effort weights must match action size")
            if any(weight < 0.0 for weight in action_effort_weights):
                raise ValueError("action effort weights cannot be negative")
            effort_weights = torch.tensor(
                action_effort_weights,
                device=starts.deterministic.device,
                dtype=starts.deterministic.dtype,
            )
        state = starts.detached()
        log_probs = []
        entropies = []
        means = []
        rewards = []
        continues = []
        features = []
        target_values = []
        efforts = []
        for _ in range(horizon):
            feature = state.features.detach()
            distribution = self.actor_distribution(feature)
            if self.actor_distribution_mode == "dreamerv3_bounded_normal":
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
            else:
                raw_action = distribution.sample()
                action = torch.tanh(raw_action)
                log_prob = distribution.log_prob(raw_action)
                log_prob -= torch.log1p(-action.square() + 1e-6)
            with torch.no_grad():
                next_state = self.rssm.imagine_step(state, action)
                next_feature = next_state.features
                if reward_source == "informed_decoder_progress":
                    predicted_reward = self.decoded_progress_reward(
                        feature, next_feature
                    )
                elif reward_source == "informed_decoder_skydreamer":
                    predicted_reward = self.decoded_skydreamer_reward(
                        feature,
                        next_feature,
                        control_frequency_hz=control_frequency_hz,
                    )
                else:
                    predicted_reward = self.predict_reward(next_feature, action)
                if effort_weights is None:
                    effort = torch.zeros_like(predicted_reward)
                else:
                    effort = (action.square() * effort_weights).sum(-1)
                    predicted_reward = predicted_reward - effort
                predicted_continue = torch.sigmoid(
                    self.continue_predictor(next_feature).squeeze(-1)
                )
                target_value = symexp(
                    self.target_critic(next_feature).squeeze(-1)
                )
            features.append(feature)
            log_probs.append(log_prob.sum(-1))
            entropies.append(distribution.entropy().sum(-1))
            means.append(self.deterministic_actor_action(distribution))
            rewards.append(predicted_reward)
            continues.append(predicted_continue)
            target_values.append(target_value)
            efforts.append(effort)
            state = next_state.detached()

        reward_tensor = torch.stack(rewards)
        continue_tensor = torch.stack(continues)
        target_value_tensor = torch.stack(target_values)
        returns = torch.empty_like(reward_tensor)
        next_return = target_value_tensor[-1]
        for step in range(horizon - 1, -1, -1):
            next_value = target_value_tensor[step]
            bootstrap = (1.0 - lambda_) * next_value + lambda_ * next_return
            next_return = reward_tensor[step] + gamma * continue_tensor[step] * bootstrap
            returns[step] = next_return

        feature_tensor = torch.stack(features)
        critic_symlog = self.critic(feature_tensor).squeeze(-1)
        # The pinned DreamerV3 actor uses the slow/target value baseline and
        # separately normalizes advantages. The former local approximation
        # used the online critic and clamped a return inter-quantile range to
        # at least one. VQ2's useful action differences are ~1e-2, so that
        # clamp erased the paper's variance normalization and left REINFORCE
        # dominated by stochastic-latent noise (N579).
        advantage = returns - target_value_tensor
        if advantage_normalization == "dreamerv3_percentile":
            with torch.no_grad():
                if update_return_normalizer:
                    low = torch.quantile(returns.detach().float(), 0.05)
                    high = torch.quantile(returns.detach().float(), 0.95)
                    self.return_normalizer_low.mul_(0.99).add_(0.01 * low)
                    self.return_normalizer_high.mul_(0.99).add_(0.01 * high)
                advantage_scale = (
                    self.return_normalizer_high - self.return_normalizer_low
                ).clamp_min(1.0)
            normalized_advantage = advantage / advantage_scale
        else:
            advantage_offset = advantage.detach().mean()
            advantage_scale = advantage.detach().std(unbiased=False).clamp_min(1e-4)
            normalized_advantage = (
                advantage - advantage_offset
            ) / advantage_scale
        log_prob_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        # The pinned DreamerV3 loss discounts policy and value terms by the
        # cumulative predicted continuation of each imagined trajectory.  This
        # is separate from lambda-return discounting: without it, uncertain
        # late prior states contribute the same gradient mass as supported
        # early states and can dominate weak CTBR channels (N589).
        continuation_weight = torch.cumprod(
            gamma * continue_tensor.detach(), dim=0
        ) / gamma
        reinforce_loss = -(
            continuation_weight
            * log_prob_tensor
            * normalized_advantage.detach()
        ).mean()
        entropy_loss = -entropy_weight * (
            continuation_weight * entropy_tensor
        ).mean()
        mean_tensor = torch.stack(means)
        smoothness = (mean_tensor[1:] - mean_tensor[:-1]).square().sum(-1).mean()
        actor_loss = reinforce_loss + entropy_loss + smoothness_weight * smoothness
        critic_error = F.smooth_l1_loss(
            critic_symlog, symlog(returns.detach()), reduction="none"
        )
        critic_loss = (continuation_weight * critic_error).mean()
        return ImaginationLoss(
            actor=actor_loss,
            critic=critic_loss,
            reinforce=reinforce_loss,
            entropy=entropy_tensor.mean(),
            smoothness=smoothness,
            mean_effort=torch.stack(efforts).mean(),
            mean_return=returns.mean(),
            return_scale=advantage_scale,
            mean_continuation_weight=continuation_weight.mean(),
        )

    @torch.no_grad()
    def update_target_critic(self, rate: float = 0.02) -> None:
        rate = float(max(0.0, min(1.0, rate)))
        for target, source in zip(
            self.target_critic.parameters(), self.critic.parameters(), strict=True
        ):
            target.lerp_(source, rate)
