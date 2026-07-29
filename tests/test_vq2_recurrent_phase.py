from __future__ import annotations

import pytest
import torch

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent import VQ2RecurrentActor
from pufferlib.vq2_recurrent_phase import (
    PHASE_LEGAL_OBS_SIZE,
    VQ2PhaseRecurrentActor,
)


def test_phase_zero_transfer_is_base_actor_exact() -> None:
    torch.manual_seed(39)
    base = VQ2RecurrentActor(hidden_size=32)
    phase = VQ2PhaseRecurrentActor(hidden_size=32)
    phase.load_phase_zero_base_state(base.state_dict())
    base.eval()
    phase.eval()
    legal = torch.randn(3, 5, LEGAL_OBS_SIZE)
    observation = torch.cat((legal, torch.zeros(3, 5, 1)), -1)
    base_output, base_state = base.forward_sequence(legal)
    phase_output, phase_state = phase.forward_sequence(observation)
    assert torch.equal(base_output.mean, phase_output.mean)
    assert torch.equal(base_output.pre_tanh_mean, phase_output.pre_tanh_mean)
    assert torch.equal(base_state, phase_state)


def test_phase_actor_rejects_wrong_width_and_invalid_phase() -> None:
    actor = VQ2PhaseRecurrentActor(hidden_size=16)
    with pytest.raises(ValueError, match="4119"):
        actor.forward_step(torch.zeros(1, PHASE_LEGAL_OBS_SIZE - 1))
    observation = torch.zeros(1, PHASE_LEGAL_OBS_SIZE)
    observation[:, -1] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        actor.forward_step(observation)


def test_phase_actor_still_has_one_joint_four_channel_head() -> None:
    actor = VQ2PhaseRecurrentActor(hidden_size=32)
    assert actor.action_head.out_features == 4
    assert actor.recurrent.num_layers == 1
    assert actor.phase_embedding.bias is None

