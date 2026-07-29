from __future__ import annotations

import torch

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent import VQ2RecurrentActor
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseResidualActor


def test_phase_zero_is_base_exact_after_residual_weights_change() -> None:
    torch.manual_seed(41)
    base = VQ2RecurrentActor(hidden_size=32)
    actor = VQ2PhaseResidualActor(hidden_size=32)
    actor.load_phase_zero_base_state(base.state_dict())
    with torch.no_grad():
        actor.phase_embedding.weight.normal_()
        actor.phase_action_residual.weight.normal_()
    legal = torch.randn(3, 5, LEGAL_OBS_SIZE)
    phase_observation = torch.cat((legal, torch.zeros(3, 5, 1)), -1)
    base_output, base_state = base.forward_sequence(legal)
    actor_output, actor_state = actor.forward_sequence(phase_observation)
    assert torch.equal(base_output.mean, actor_output.mean)
    assert torch.equal(base_state, actor_state)


def test_phase_residual_remains_one_actor_and_joint_vector() -> None:
    actor = VQ2PhaseResidualActor(hidden_size=32)
    observation = torch.zeros(2, PHASE_LEGAL_OBS_SIZE)
    observation[:, -1] = 1.0 / 6.0
    output, state = actor.forward_step(observation)
    assert output.mean.shape == (2, 4)
    assert state.shape == (1, 2, 32)
    assert actor.recurrent.num_layers == 1

