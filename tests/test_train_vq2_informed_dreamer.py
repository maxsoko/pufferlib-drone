from __future__ import annotations

import copy
import math
import sys

import pytest
import torch

from pufferlib.vq2_dreamer import VQ2InformedDreamer
from pufferlib.vq2_informed import ENV_OBS_SIZE, MASK_SIZE
from pufferlib.vq2_informed import configure_full_start_collection
from scripts.train_vq2_informed_dreamer import (
    _apply_collector_action_hold,
    _autocast_context,
    _load_config,
    _tensor_from_pointer,
    parse_args,
    _world_model_update,
)


def _model() -> VQ2InformedDreamer:
    torch.manual_seed(31)
    return VQ2InformedDreamer(
        deterministic_size=32,
        stochastic_groups=4,
        stochastic_classes=4,
    )


def _batch(
    batch_size: int,
    sequence_length: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(47)
    observation = torch.randn(
        batch_size, sequence_length, ENV_OBS_SIZE, generator=generator
    )
    observation[..., :MASK_SIZE] = torch.rand(
        batch_size, sequence_length, MASK_SIZE, generator=generator
    )
    action = torch.tanh(
        torch.randn(batch_size, sequence_length, 4, generator=generator)
    )
    reward = torch.randn(batch_size, sequence_length, generator=generator)
    continuation = torch.ones(batch_size, sequence_length)
    return tuple(
        value.to(device)
        for value in (observation, action, reward, continuation)
    )


def _world_parameters(model: VQ2InformedDreamer) -> list[torch.nn.Parameter]:
    return [
        *model.rssm.parameters(),
        *model.privileged_decoder.parameters(),
        *model.reward_predictor.parameters(),
        *model.continue_predictor.parameters(),
    ]


def test_weighted_microbatch_matches_full_logical_batch() -> None:
    full_model = _model()
    split_model = copy.deepcopy(full_model)
    full_parameters = _world_parameters(full_model)
    split_parameters = _world_parameters(split_model)
    full_optimizer = torch.optim.SGD(full_parameters, lr=1e-3)
    split_optimizer = torch.optim.SGD(split_parameters, lr=1e-3)
    batch = _batch(3, 5)

    full_loss, full_start, full_norm, full_count = _world_model_update(
        model=full_model,
        optimizer=full_optimizer,
        observation=batch[0],
        action=batch[1],
        reward=batch[2],
        continuation=batch[3],
        replay_context=2,
        microbatch_size=3,
        amp_dtype="none",
        world_parameters=full_parameters,
        deterministic_latent=True,
    )
    split_loss, split_start, split_norm, split_count = _world_model_update(
        model=split_model,
        optimizer=split_optimizer,
        observation=batch[0],
        action=batch[1],
        reward=batch[2],
        continuation=batch[3],
        replay_context=2,
        microbatch_size=2,
        amp_dtype="none",
        world_parameters=split_parameters,
        deterministic_latent=True,
    )

    assert full_count == 1
    assert split_count == 2
    for name in full_loss.__dataclass_fields__:
        assert torch.allclose(
            getattr(full_loss, name), getattr(split_loss, name), atol=2e-6, rtol=2e-6
        )
    for full_value, split_value in zip(full_start, split_start, strict=True):
        assert torch.allclose(full_value, split_value, atol=2e-6, rtol=2e-6)
    assert torch.allclose(full_norm, split_norm, atol=2e-5, rtol=2e-5)
    for full_parameter, split_parameter in zip(
        full_parameters, split_parameters, strict=True
    ):
        assert torch.allclose(
            full_parameter, split_parameter, atol=2e-6, rtol=2e-6
        )


def test_bfloat16_amp_rejects_cpu() -> None:
    with pytest.raises(RuntimeError, match="only on CUDA"):
        with _autocast_context(torch.device("cpu"), "bfloat16"):
            pass


def test_discovery_defaults_match_paper_temporal_and_entropy_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train_vq2_informed_dreamer.py"])
    args = parse_args()
    assert args.replay_context == 16
    assert args.full_start_collection is False
    assert args.world_sequence_length == 64
    assert args.imagination_horizon == 16
    assert args.gamma == pytest.approx(0.997)
    assert args.entropy_weight == pytest.approx(3e-4)
    assert args.smoothness_weight == pytest.approx(0.002)
    assert args.collector_policy_sampling is True
    assert args.actor_initial_std == pytest.approx(0.20)
    assert args.actor_distribution_mode == "dreamerv3_bounded_normal"
    assert args.advantage_normalization == "dreamerv3_percentile"
    assert args.weight_decay == pytest.approx(0.0)
    assert args.collector_action_hold == 1
    assert args.free_nats == pytest.approx(1.0)
    assert args.distributional_reward is True
    assert args.action_conditioned_reward is False
    assert args.imagination_reward_source == "informed_decoder_skydreamer"
    assert args.imagination_start_mode == "all"
    assert args.imagination_start_count == 0
    assert args.action_effort_weights == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert args.resume_refresh_steps == 128
    assert args.reset_actor_optimizer_on_resume is False
    assert args.reset_critic_optimizer_on_resume is False


def test_full_start_collection_disables_every_local_curriculum() -> None:
    config = _load_config("drone_race_vq2_informed_dreamer")
    configured = configure_full_start_collection(config)
    environment = configured["env"]
    assert environment["gate_local_start_curriculum"] == 0
    assert environment["gate_local_start_probability"] == 0.0
    assert environment["mixed_start_curriculum"] == 0
    assert environment["segment_start_probability"] == 0.0
    assert environment["evaluation_episode_limit"] == 0


def test_world_update_can_return_bounded_all_time_imagination_starts() -> None:
    model = _model()
    parameters = _world_parameters(model)
    optimizer = torch.optim.SGD(parameters, lr=1e-3)
    batch = _batch(3, 5)
    _, starts, _, _ = _world_model_update(
        model=model,
        optimizer=optimizer,
        observation=batch[0],
        action=batch[1],
        reward=batch[2],
        continuation=batch[3],
        replay_context=2,
        microbatch_size=2,
        amp_dtype="none",
        world_parameters=parameters,
        deterministic_latent=True,
        imagination_start_mode="all",
        imagination_start_count=5,
    )
    assert starts.deterministic.shape == (5, 32)
    assert starts.stochastic.shape == (5, 4, 4)
    assert starts.logits.shape == (5, 4, 4)


def test_collector_action_hold_resamples_only_on_expiry_or_reset() -> None:
    held = torch.zeros(2, 4)
    remaining = torch.zeros(2, dtype=torch.long)
    first = torch.tensor([[0.2, 0.1, 0.0, -0.1], [-0.2, 0.0, 0.1, 0.2]])
    action, held, remaining = _apply_collector_action_hold(
        first, held, remaining, 3
    )
    assert torch.equal(action, first)
    assert torch.equal(remaining, torch.tensor([2, 2]))

    ignored = torch.full((2, 4), 0.9)
    for expected in (1, 0):
        action, held, remaining = _apply_collector_action_hold(
            ignored, held, remaining, 3
        )
        assert torch.equal(action, first)
        assert torch.equal(remaining, torch.tensor([expected, expected]))

    reset = torch.tensor([True, False])
    held = torch.where(reset[:, None], torch.zeros_like(held), held)
    remaining = torch.where(reset, torch.zeros_like(remaining), remaining)
    replacement = torch.tensor([[0.3, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0]])
    action, _, remaining = _apply_collector_action_hold(
        replacement, held, remaining, 3
    )
    assert torch.equal(action, replacement)
    assert torch.equal(remaining, torch.tensor([2, 2]))


def test_reset_sampler_log_includes_still_running_vector_environments() -> None:
    from pufferlib import _C

    if getattr(_C, "env_name", None) != "drone_race_vision":
        pytest.skip("drone_race_vision native binding is not loaded")
    config = _load_config("drone_race_vq2_informed_dreamer")
    config["env"].update(
        {
            "pos_bound": 10.0,
            "gate_position_domain_randomize": 0,
            "course_geometry_scale_randomize": 0,
            "reset_position_noise_xy": 0.0,
            "reset_position_noise_z": 0.0,
        }
    )
    vec = _C.create_vec(config, gpu=0)
    actions = torch.zeros((vec.total_agents, vec.num_atns), dtype=torch.float32)
    try:
        vec.reset()
        vec.cpu_step(actions.data_ptr())
        log = dict(vec.log())
    finally:
        vec.close()

    completed = float(log["n"])
    assert 0.0 < completed < float(config["vec"]["total_agents"])
    raw_resets = float(log["reset_count"]) * completed
    raw_local_resets = float(log["gate_local_reset_count"]) * completed
    assert raw_resets == pytest.approx(
        float(config["vec"]["total_agents"]), abs=1e-5
    )
    assert 0.0 < raw_local_resets < raw_resets
    assert raw_local_resets / raw_resets == pytest.approx(
        float(log["gate_local_reset_count"]) / float(log["reset_count"])
    )


def test_discovery_resets_begin_above_configured_ground_plane() -> None:
    from pufferlib import _C
    from pufferlib.vq2_informed import LEGAL_OBS_SIZE

    if getattr(_C, "env_name", None) != "drone_race_vision":
        pytest.skip("drone_race_vision native binding is not loaded")
    config = _load_config("drone_race_vq2_informed_dreamer")
    vec = _C.create_vec(config, gpu=0)
    try:
        vec.reset()
        observations = _tensor_from_pointer(
            vec.obs_ptr, vec.total_agents * vec.obs_size
        ).reshape(vec.total_agents, vec.obs_size).clone()
    finally:
        vec.close()

    normalized_ground_z = math.tanh(float(config["env"]["crash_height"]) / 20.0)
    assert torch.all(observations[:, LEGAL_OBS_SIZE + 2] > normalized_ground_z)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_microbatch_update_is_finite() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    device = torch.device("cuda")
    model = _model().to(device)
    parameters = _world_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=4e-5)
    batch = _batch(2, 4, device=device)
    loss, start, norm, count = _world_model_update(
        model=model,
        optimizer=optimizer,
        observation=batch[0],
        action=batch[1],
        reward=batch[2],
        continuation=batch[3],
        replay_context=1,
        microbatch_size=1,
        amp_dtype="bfloat16",
        world_parameters=parameters,
    )
    assert count == 2
    assert all(torch.isfinite(getattr(loss, name)) for name in loss.__dataclass_fields__)
    assert all(torch.isfinite(value).all() for value in start)
    assert torch.isfinite(norm)
