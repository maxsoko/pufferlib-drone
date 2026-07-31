from __future__ import annotations

import numpy as np
import torch

from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseMLPResidualActor
from scripts.train_vq2_vg068_indexed_mlp_intervention_features import (
    CONFIG,
    TARGET_PHASES,
    non_target_outputs_zero,
    phase_action_prediction,
    validation_mask,
)


def test_group_stratified_validation_covers_all_count_residues() -> None:
    agents = np.arange(512, dtype=np.uint16)
    held = agents[validation_mask(agents)]
    assert held.size == 64
    assert np.bincount(held % 8, minlength=8).tolist() == [8] * 8


def test_phase_prediction_is_base_exact_at_zero_output() -> None:
    torch.manual_seed(9)
    model = VQ2IndexedPhaseMLPResidualActor(hidden_size=16, residual_size=8)
    hidden = torch.randn(7, 16)
    base = torch.randn(7, 4)
    prediction = phase_action_prediction(model, hidden, base, 3)
    torch.testing.assert_close(prediction, torch.tanh(base), rtol=0, atol=0)


def test_phase_prediction_trains_only_selected_public_head() -> None:
    torch.manual_seed(10)
    model = VQ2IndexedPhaseMLPResidualActor(hidden_size=16, residual_size=8)
    hidden = torch.randn(7, 16)
    base = torch.randn(7, 4)
    with torch.no_grad():
        model.indexed_phase_residual_output[2].fill_(0.1)
    phase2 = phase_action_prediction(model, hidden, base, 2)
    phase3 = phase_action_prediction(model, hidden, base, 3)
    assert not torch.equal(phase2, torch.tanh(base))
    torch.testing.assert_close(phase3, torch.tanh(base), rtol=0, atol=0)


def test_non_target_outputs_must_remain_zero() -> None:
    model = VQ2IndexedPhaseMLPResidualActor(hidden_size=16, residual_size=8)
    assert non_target_outputs_zero(model)
    with torch.no_grad():
        model.indexed_phase_residual_output_bias[12, 0] = 0.1
    assert not non_target_outputs_zero(model)


def test_fixed_training_contract() -> None:
    assert TARGET_PHASES == tuple(range(1, 12))
    assert CONFIG.epochs == 10
    assert CONFIG.residual_size == 64
    assert CONFIG.minimum_aggregate_improvement == 2.0
