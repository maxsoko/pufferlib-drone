from __future__ import annotations

import torch

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase_residual import (
    VQ2PhaseLocalAdapterActor,
    VQ2StackedPhaseRangeAdapterActor,
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


def test_stacked_zero_continuation_is_exact_phase_local_parent() -> None:
    torch.manual_seed(9)
    parent = VQ2PhaseLocalAdapterActor(target_phase=15)
    stacked = VQ2StackedPhaseRangeAdapterActor(
        target_phase=15,
        continuation_phase_min=16,
        continuation_phase_max_exclusive=18,
    )
    stacked.load_phase_local_state(parent.state_dict())
    observation = torch.randn(5, PHASE_LEGAL_OBS_SIZE)
    observation[:, LEGAL_OBS_SIZE] = torch.tensor([15, 16, 17, 18, 14]) / 6.0
    parent_state = parent.initial_state(5, device="cpu")
    stacked_state = stacked.initial_state(5, device="cpu")
    with torch.no_grad():
        parent_output, parent_next = parent.forward_step(observation, parent_state)
        stacked_output, stacked_next = stacked.forward_step(observation, stacked_state)
    assert torch.equal(parent_output.mean, stacked_output.mean)
    assert torch.equal(parent_output.pre_tanh_mean, stacked_output.pre_tanh_mean)
    assert torch.equal(parent_next, stacked_next[..., : parent_next.shape[-1]])
    continuation = stacked_next[..., parent_next.shape[-1] :]
    assert torch.count_nonzero(continuation[:, [0, 3, 4]]) == 0
    assert torch.count_nonzero(continuation[:, [1, 2]]) > 0


def test_stacked_sequence_matches_repeated_steps() -> None:
    torch.manual_seed(10)
    actor = VQ2StackedPhaseRangeAdapterActor(
        target_phase=15,
        continuation_phase_min=16,
        continuation_phase_max_exclusive=18,
    )
    observation = torch.randn(3, 4, PHASE_LEGAL_OBS_SIZE)
    observation[..., LEGAL_OBS_SIZE] = 16.0 / 6.0
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
