from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.train_vq2_recurrent_bc import weighted_action_mse
import scripts.train_vq2_variable_gate_four_source_refit as refit


def test_four_source_loss_uses_fixed_weights_despite_record_counts() -> None:
    config = replace(refit.FourSourceConfig(), smoothness_weight=0.0)
    weights = torch.tensor(config.action_weights)
    lengths = {"clean": 1, "dagger1": 5, "dagger2": 9, "dagger3": 13}
    predictions = {
        name: torch.zeros((1, length, ACTION_SIZE))
        for name, length in lengths.items()
    }
    targets = {
        name: torch.full_like(predictions[name], float(index + 1))
        for index, name in enumerate(lengths)
    }
    valid = {
        name: torch.ones((1, length), dtype=torch.bool)
        for name, length in lengths.items()
    }
    loss, components = refit.four_source_loss(
        predictions,
        targets,
        valid,
        weights=weights,
        previous_predictions={name: None for name in lengths},
        previous_valid={name: None for name in lengths},
        config=config,
    )
    expected = 0.0
    for name, source_weight in refit.SOURCE_WEIGHTS.items():
        source_loss, _ = weighted_action_mse(
            predictions[name], targets[name], valid[name], weights
        )
        expected = expected + source_weight * source_loss
        assert torch.equal(components[f"{name}_total"], source_loss)
    assert torch.equal(loss, expected)


def test_vg017_configuration_and_source_weight_audit_are_exact() -> None:
    config = refit.FourSourceConfig()
    assert config.sequence_chunk >= 256
    assert config.transition_window_exposure >= 3
    assert config.learning_rate == pytest.approx(2e-5)
    assert refit.objective_weights(config) == {
        "clean": 0.40,
        "dagger1": 0.15,
        "dagger2": 0.15,
        "dagger3": 0.30,
    }
    assert refit.source_weight_audit(
        {"clean": 4.0, "dagger1": 1.5, "dagger2": 1.5, "dagger3": 3.0},
        10,
        config,
    )
    assert not refit.source_weight_audit(
        {"clean": 4.0, "dagger1": 1.0, "dagger2": 2.0, "dagger3": 3.0},
        10,
        config,
    )


def test_vg017_frozen_inputs_verify() -> None:
    refit.verify_inputs()


def test_vg017_runner_is_source_locked_resumable_and_offline() -> None:
    text = refit.RUNNER.read_text()
    assert os.access(refit.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git status --porcelain --untracked-files=no" in text
    assert refit.TAG in text
    assert "tests/test_train_vq2_variable_gate_four_source_refit.py" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
