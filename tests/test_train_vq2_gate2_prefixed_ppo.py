import torch

from scripts.policy_callable_checkpoint import CheckpointPolicy
from scripts.train_full_policy_bc import SequencePufferNet
from scripts.train_vq2_gate2_prefixed_ppo import (
    compute_gae,
    gate2_initial_state,
    mask_policy_gradients,
    reset_recurrent_state,
)


def test_compute_gae_stops_bootstrap_at_terminal():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.25]])
    dones = torch.tensor([[0.0], [1.0]])

    advantages, returns = compute_gae(
        rewards, values, dones, torch.tensor([99.0]), gamma=1.0, gae_lambda=1.0)

    torch.testing.assert_close(advantages, torch.tensor([[2.5], [1.75]]))
    torch.testing.assert_close(returns, torch.tensor([[3.0], [2.0]]))


def test_reset_recurrent_state_replaces_only_done_agents():
    state = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    reset = torch.full_like(state, -1.0)

    result = reset_recurrent_state(state, torch.tensor([False, True]), reset)

    torch.testing.assert_close(result[:, 0], state[:, 0])
    torch.testing.assert_close(result[:, 1], reset[:, 1])


def test_compute_gae_ignores_bootstrap_after_inactive_padding():
    rewards = torch.tensor([[1.0], [0.0], [0.0]])
    values = torch.tensor([[0.5], [7.0], [9.0]])
    dones = torch.tensor([[1.0], [1.0], [1.0]])

    advantages, _ = compute_gae(
        rewards, values, dones, torch.tensor([11.0]), gamma=0.99, gae_lambda=0.95)

    assert advantages[0, 0] == 0.5


def test_mask_policy_gradients_preserves_only_requested_action_rows():
    checkpoint = CheckpointPolicy(
        input_dim=2,
        hidden_dim=3,
        num_actions=4,
        num_layers=1,
        layout_precision_bytes=4,
        encoder=torch.zeros(3, 2).numpy(),
        decoder=torch.zeros(5, 3).numpy(),
        log_std=torch.zeros(4).numpy(),
        mingru_proj=[torch.zeros(9, 3).numpy()],
        state=torch.zeros(1, 3).numpy(),
    )
    model = SequencePufferNet(checkpoint)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    mask_policy_gradients(model, (2,))

    assert model.encoder.grad is None
    assert model.mingru[0].grad is None
    torch.testing.assert_close(model.decoder.grad[2], torch.ones(3))
    frozen_rows = torch.cat((model.decoder.grad[:2], model.decoder.grad[3:]))
    torch.testing.assert_close(frozen_rows, torch.zeros_like(frozen_rows))


def test_mask_policy_gradients_noop_without_row_selection():
    checkpoint = CheckpointPolicy(
        input_dim=2,
        hidden_dim=3,
        num_actions=4,
        num_layers=1,
        layout_precision_bytes=4,
        encoder=torch.zeros(3, 2).numpy(),
        decoder=torch.zeros(5, 3).numpy(),
        log_std=torch.zeros(4).numpy(),
        mingru_proj=[torch.zeros(9, 3).numpy()],
        state=torch.zeros(1, 3).numpy(),
    )
    model = SequencePufferNet(checkpoint)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    mask_policy_gradients(model, None)

    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))


def test_gate2_initial_state_can_reset_instead_of_warming_prefix():
    checkpoint = CheckpointPolicy(
        input_dim=2,
        hidden_dim=3,
        num_actions=4,
        num_layers=1,
        layout_precision_bytes=4,
        encoder=torch.ones(3, 2).numpy(),
        decoder=torch.zeros(5, 3).numpy(),
        log_std=torch.zeros(4).numpy(),
        mingru_proj=[torch.ones(9, 3).numpy()],
        state=torch.zeros(1, 3).numpy(),
    )
    model = SequencePufferNet(checkpoint)
    prefix = torch.ones(4, 2)

    reset = gate2_initial_state(
        model, prefix, 2, reset_recurrent_at_gate2=True)
    warmed = gate2_initial_state(
        model, prefix, 2, reset_recurrent_at_gate2=False)

    torch.testing.assert_close(reset, torch.zeros_like(reset))
    assert torch.count_nonzero(warmed) > 0
