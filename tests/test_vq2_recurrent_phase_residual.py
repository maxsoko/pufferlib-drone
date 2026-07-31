from __future__ import annotations

import torch

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent import VQ2RecurrentActor
from pufferlib.vq2_recurrent_phase import (
    PHASE_LEGAL_OBS_SIZE,
    VQ2PhaseRecurrentActor,
)
from pufferlib.vq2_recurrent_phase_residual import (
    VQ2IndexedPhaseMLPResidualActor,
    VQ2IndexedPhaseResidualActor,
    VQ2PhaseResidualActor,
    VQ2UnboundedProgressMLPResidualActor,
)


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


def test_indexed_residual_zero_is_base_exact() -> None:
    torch.manual_seed(7)
    base = VQ2PhaseRecurrentActor(hidden_size=32)
    indexed = VQ2IndexedPhaseResidualActor(hidden_size=32)
    indexed.load_base_state(base.state_dict())
    observation = torch.randn(2, 5, PHASE_LEGAL_OBS_SIZE)
    observation[..., -1] = torch.arange(5).float() / 16.0
    with torch.no_grad():
        base_output, base_state = base.forward_sequence(observation)
        indexed_output, indexed_state = indexed.forward_sequence(observation)
    assert torch.equal(base_output.mean, indexed_output.mean)
    assert torch.equal(base_state, indexed_state)


def test_indexed_residual_changes_only_selected_phase() -> None:
    torch.manual_seed(8)
    base = VQ2PhaseRecurrentActor(hidden_size=32)
    indexed = VQ2IndexedPhaseResidualActor(hidden_size=32)
    indexed.load_base_state(base.state_dict())
    with torch.no_grad():
        indexed.indexed_phase_action_residual[4].fill_(0.1)
    observation = torch.randn(1, 2, PHASE_LEGAL_OBS_SIZE)
    observation[..., -1] = torch.tensor([3, 4]).float() / 16.0
    with torch.no_grad():
        base_output, _ = base.forward_sequence(observation)
        indexed_output, _ = indexed.forward_sequence(observation)
    assert torch.equal(base_output.mean[:, 0], indexed_output.mean[:, 0])
    assert not torch.equal(base_output.mean[:, 1], indexed_output.mean[:, 1])


def test_indexed_phase_mlp_zero_output_is_base_exact() -> None:
    torch.manual_seed(31)
    base = VQ2PhaseRecurrentActor(hidden_size=32)
    actor = VQ2IndexedPhaseMLPResidualActor(hidden_size=32, residual_size=8)
    actor.load_base_state(base.state_dict())
    observation = torch.randn(4, PHASE_LEGAL_OBS_SIZE)
    observation[:, -1] = torch.tensor([0.0, 1.0 / 16.0, 4.0 / 16.0, 11.0 / 16.0])
    state = base.initial_state(4, device="cpu")
    expected, expected_state = base.forward_step(observation, state)
    actual, actual_state = actor.forward_step(observation, state)
    torch.testing.assert_close(actual.mean, expected.mean, rtol=0, atol=0)
    torch.testing.assert_close(actual.pre_tanh_mean, expected.pre_tanh_mean, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


def test_indexed_phase_mlp_changes_only_selected_head() -> None:
    torch.manual_seed(37)
    actor = VQ2IndexedPhaseMLPResidualActor(hidden_size=32, residual_size=8)
    observation = torch.randn(2, PHASE_LEGAL_OBS_SIZE)
    observation[:, -1] = torch.tensor([2.0 / 16.0, 3.0 / 16.0])
    state = actor.initial_state(2, device="cpu")
    baseline, _ = actor.forward_step(observation, state)
    with torch.no_grad():
        actor.indexed_phase_residual_output[2].fill_(0.2)
        actor.indexed_phase_residual_output_bias[2].fill_(0.1)
    changed, _ = actor.forward_step(observation, state)
    assert not torch.equal(changed.mean[0], baseline.mean[0])
    torch.testing.assert_close(changed.mean[1], baseline.mean[1], rtol=0, atol=0)


def test_unbounded_conversion_preserves_legacy_indices() -> None:
    torch.manual_seed(43)
    legacy = VQ2IndexedPhaseMLPResidualActor(hidden_size=32, residual_size=8)
    with torch.no_grad():
        legacy.phase_embedding.weight.normal_(std=0.1)
        legacy.indexed_phase_residual_output.normal_(std=0.05)
        legacy.indexed_phase_residual_output_bias.normal_(std=0.02)
    converted = VQ2UnboundedProgressMLPResidualActor(
        hidden_size=32, residual_size=8
    )
    converted.load_converted_state(legacy.state_dict())
    legal = torch.randn(2, 9, LEGAL_OBS_SIZE)
    indices = torch.tensor([0, 1, 2, 4, 6, 8, 11, 15, 16], dtype=torch.float32)
    old_observation = torch.cat(
        (legal, (indices / 16.0).view(1, 9, 1).expand(2, -1, -1)), -1
    )
    new_observation = torch.cat(
        (legal, (indices / 6.0).view(1, 9, 1).expand(2, -1, -1)), -1
    )
    with torch.no_grad():
        expected, expected_state = legacy.forward_sequence(old_observation)
        actual, actual_state = converted.forward_sequence(new_observation)
    torch.testing.assert_close(actual.mean, expected.mean, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-6, atol=2e-6)


def test_unbounded_actor_tracks_twenty_and_falls_back_beyond_native_cap() -> None:
    torch.manual_seed(47)
    actor = VQ2UnboundedProgressMLPResidualActor(hidden_size=32, residual_size=8)
    baseline = VQ2UnboundedProgressMLPResidualActor(hidden_size=32, residual_size=8)
    baseline.load_state_dict(actor.state_dict())
    with torch.no_grad():
        actor.indexed_phase_residual_output_bias[20].fill_(0.2)
        actor.indexed_phase_residual_output_bias[32].fill_(0.4)
    observation = torch.zeros(2, PHASE_LEGAL_OBS_SIZE)
    observation[:, -1] = torch.tensor([20.0 / 6.0, 33.0 / 6.0])
    with torch.no_grad():
        selected, _ = actor.forward_step(observation)
        reference, _ = baseline.forward_step(observation)
    assert not torch.equal(selected.mean[0], reference.mean[0])
    torch.testing.assert_close(selected.mean[1], reference.mean[1], rtol=0, atol=0)
