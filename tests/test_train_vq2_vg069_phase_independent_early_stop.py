from __future__ import annotations

import torch

from pufferlib.vq2_recurrent_phase_residual import VQ2IndexedPhaseMLPResidualActor
from scripts.train_vq2_vg069_phase_independent_early_stop import (
    TARGET_PHASES,
    capture_phase,
    restore_phase,
    update_phase_bests,
)


def _metrics(value: float) -> dict:
    return {
        "phases": {
            str(phase): {"selected_action_mse": value + phase * 1e-5}
            for phase in TARGET_PHASES
        }
    }


def test_phase_capture_restore_changes_only_selected_head() -> None:
    torch.manual_seed(11)
    model = VQ2IndexedPhaseMLPResidualActor(hidden_size=16, residual_size=8)
    phase2 = capture_phase(model, 2)
    phase3 = capture_phase(model, 3)
    with torch.no_grad():
        for name in phase2:
            getattr(model, name)[2].add_(1.0)
            getattr(model, name)[3].add_(2.0)
    restore_phase(model, 2, phase2)
    for name, value in phase2.items():
        torch.testing.assert_close(getattr(model, name)[2], value, rtol=0, atol=0)
    for name, value in phase3.items():
        assert not torch.equal(getattr(model, name)[3], value)


def test_phase_bests_update_independently() -> None:
    model = VQ2IndexedPhaseMLPResidualActor(hidden_size=16, residual_size=8)
    objectives = {phase: 0.5 for phase in TARGET_PHASES}
    epochs = {phase: 0 for phase in TARGET_PHASES}
    states = {phase: capture_phase(model, phase) for phase in TARGET_PHASES}
    update_phase_bests(model, _metrics(0.1), 3, objectives, epochs, states)
    assert set(epochs.values()) == {3}
    prior = {phase: capture_phase(model, phase) for phase in TARGET_PHASES}
    update_phase_bests(model, _metrics(0.6), 4, objectives, epochs, states)
    assert set(epochs.values()) == {3}
    for phase in TARGET_PHASES:
        for name in prior[phase]:
            torch.testing.assert_close(states[phase][name], prior[phase][name])


def test_parent_epoch_zero_remains_available() -> None:
    model = VQ2IndexedPhaseMLPResidualActor(hidden_size=16, residual_size=8)
    objectives = {phase: 0.01 for phase in TARGET_PHASES}
    epochs = {phase: 0 for phase in TARGET_PHASES}
    states = {phase: capture_phase(model, phase) for phase in TARGET_PHASES}
    update_phase_bests(model, _metrics(0.1), 1, objectives, epochs, states)
    assert set(epochs.values()) == {0}
