import torch

from scripts.continue_vq2_phase_transition_offline import (
    PHASE_INDEX,
    _crossing_loss,
    _phase_targets,
    _row_mask,
)


def test_row_mask_selects_only_phase_output() -> None:
    mask = _row_mask(torch.Size((34, 5)), PHASE_INDEX)
    assert mask.shape == (34, 5)
    assert mask.sum().item() == 5.0
    assert mask[PHASE_INDEX].eq(1.0).all()
    assert mask[:PHASE_INDEX].eq(0.0).all()


def test_phase_targets_use_last_context_and_event_record() -> None:
    observation = torch.zeros(2, 4, 4152)
    phase_flat_index = -34 + PHASE_INDEX
    observation[:, 2, phase_flat_index] = torch.tensor([1 / 6, 2 / 6])
    observation[:, 3, phase_flat_index] = torch.tensor([2 / 6, 3 / 6])
    current, next_phase = _phase_targets(observation, context_length=3)
    assert torch.allclose(current, torch.tensor([1 / 6, 2 / 6]))
    assert torch.allclose(next_phase, torch.tensor([2 / 6, 3 / 6]))


def test_crossing_loss_prefers_separated_event_and_dense_deltas() -> None:
    target = torch.tensor([0.0, 1 / 6])
    unresolved = _crossing_loss(
        torch.tensor([0.0, 0.0]), target, logit_scale=20.0
    )
    separated = _crossing_loss(
        torch.tensor([0.0, 1 / 6]), target, logit_scale=20.0
    )
    assert separated < unresolved
