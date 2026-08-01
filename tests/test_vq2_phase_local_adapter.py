from __future__ import annotations

import torch

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import (
    VQ2PhaseLocalAdapterActor,
    VQ2UnboundedProgressMLPResidualActor,
)


def test_zero_adapter_is_exact_base_at_and_outside_target() -> None:
    torch.manual_seed(7)
    base = VQ2UnboundedProgressMLPResidualActor()
    adapter = VQ2PhaseLocalAdapterActor(target_phase=15)
    adapter.load_base_state(base.state_dict())
    observation = torch.randn(4, PHASE_LEGAL_OBS_SIZE)
    observation[:, LEGAL_OBS_SIZE] = torch.tensor([14.0, 15.0, 15.0, 16.0]) / 6.0
    base_state = base.initial_state(4, device="cpu")
    adapter_state = adapter.initial_state(4, device="cpu")
    with torch.no_grad():
        base_output, base_next = base.forward_step(observation, base_state)
        adapter_output, adapter_next = adapter.forward_step(observation, adapter_state)
    assert torch.equal(base_output.mean, adapter_output.mean)
    assert torch.equal(base_output.pre_tanh_mean, adapter_output.pre_tanh_mean)
    assert torch.equal(base_next, adapter_next[..., : base.hidden_size])
    assert torch.count_nonzero(adapter_next[:, [0, 3], base.hidden_size :]) == 0
    assert torch.count_nonzero(adapter_next[:, [1, 2], base.hidden_size :]) > 0


def test_adapter_sequence_matches_repeated_steps() -> None:
    torch.manual_seed(8)
    actor = VQ2PhaseLocalAdapterActor(target_phase=15)
    observation = torch.randn(3, 5, PHASE_LEGAL_OBS_SIZE)
    observation[..., LEGAL_OBS_SIZE] = 15.0 / 6.0
    state = actor.initial_state(3, device="cpu")
    with torch.no_grad():
        sequence, sequence_state = actor.forward_sequence(observation, state)
        step_state = state
        means = []
        for step in range(observation.shape[1]):
            output, step_state = actor.forward_step(observation[:, step], step_state)
            means.append(output.mean)
    assert torch.equal(sequence.mean, torch.stack(means, dim=1))
    assert torch.equal(sequence_state, step_state)


def test_adapter_split_states_are_contiguous() -> None:
    actor = VQ2PhaseLocalAdapterActor(target_phase=15)
    state = actor.initial_state(5, device="cpu")
    base = state[..., : actor.hidden_size].contiguous()
    adapter = state[0, :, actor.hidden_size :].contiguous()
    assert base.is_contiguous()
    assert adapter.is_contiguous()
