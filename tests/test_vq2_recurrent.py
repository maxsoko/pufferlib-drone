from __future__ import annotations

import pytest
import torch

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE, VQ2RecurrentActor


def test_actor_is_one_legal_cnn_gru_and_joint_action_head() -> None:
    model = VQ2RecurrentActor(hidden_size=32)
    assert model.encoder.output_size == 32
    assert model.recurrent.input_size == 32
    assert model.recurrent.hidden_size == 32
    assert model.recurrent.num_layers == 1
    assert model.action_head.in_features == 32
    assert model.action_head.out_features == ACTION_SIZE
    assert not hasattr(model, "critic")
    assert not hasattr(model, "teacher")


def test_sequence_and_causal_step_replay_are_exact_in_eval_mode() -> None:
    torch.manual_seed(7)
    model = VQ2RecurrentActor(hidden_size=32).eval()
    legal = torch.randn(2, 4, LEGAL_OBS_SIZE)
    with torch.no_grad():
        sequence, sequence_state = model.forward_sequence(legal)
        state = model.initial_state(2, device=legal.device)
        step_means = []
        for step in range(legal.shape[1]):
            output, state = model.forward_step(legal[:, step], state)
            step_means.append(output.mean)
    replay = torch.stack(step_means, 1)
    assert torch.allclose(replay, sequence.mean, atol=2e-7, rtol=0.0)
    assert torch.allclose(state, sequence_state, atol=2e-7, rtol=0.0)
    assert torch.all(sequence.mean >= -1.0)
    assert torch.all(sequence.mean <= 1.0)


def test_actor_structurally_rejects_native_privileged_tail() -> None:
    model = VQ2RecurrentActor(hidden_size=32)
    with pytest.raises(ValueError, match="4118"):
        model.forward_step(torch.zeros(1, ENV_OBS_SIZE))
    with pytest.raises(ValueError, match="4118"):
        model.forward_sequence(torch.zeros(1, 2, ENV_OBS_SIZE))


def test_sampled_gaussian_action_is_joint_and_bounded() -> None:
    model = VQ2RecurrentActor(hidden_size=32)
    output, _ = model.forward_step(torch.zeros(3, LEGAL_OBS_SIZE))
    sampled = model.sample_action(output)
    assert sampled.shape == (3, ACTION_SIZE)
    assert torch.all(sampled > -1.0)
    assert torch.all(sampled < 1.0)


def test_initial_state_contract() -> None:
    model = VQ2RecurrentActor(hidden_size=32)
    state = model.initial_state(5, device="cpu")
    assert state.shape == (1, 5, 32)
    assert torch.count_nonzero(state) == 0
    with pytest.raises(ValueError, match="batch size"):
        model.initial_state(0, device="cpu")
