from __future__ import annotations

import inspect
import os
from dataclasses import replace

import pytest
import torch

from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.train_vq2_recurrent_bc import weighted_action_mse
from scripts.train_vq2_variable_gate_recurrent_bc import VariableGateBCDataset
import scripts.train_vq2_variable_gate_three_source_refit as refit


def test_three_source_loss_uses_fixed_weights_despite_record_counts() -> None:
    config = replace(refit.ThreeSourceConfig(), smoothness_weight=0.0)
    weights = torch.tensor(config.action_weights)
    lengths = {"clean": 1, "dagger1": 5, "dagger2": 9}
    predictions = {
        name: torch.zeros((1, length, ACTION_SIZE))
        for name, length in lengths.items()
    }
    targets = {
        "clean": torch.ones_like(predictions["clean"]),
        "dagger1": torch.full_like(predictions["dagger1"], 2.0),
        "dagger2": torch.full_like(predictions["dagger2"], 3.0),
    }
    valid = {
        name: torch.ones((1, length), dtype=torch.bool)
        for name, length in lengths.items()
    }
    loss, components = refit.three_source_loss(
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


def test_vg014_configuration_and_source_weight_audit_are_exact() -> None:
    config = refit.ThreeSourceConfig()
    assert config.sequence_chunk >= 256
    assert config.transition_window_exposure >= 3
    assert config.learning_rate == pytest.approx(2e-5)
    assert (
        config.clean_objective_weight,
        config.dagger1_objective_weight,
        config.dagger2_objective_weight,
    ) == (0.5, 0.25, 0.25)
    assert refit.source_weight_audit(
        {"clean": 5.0, "dagger1": 2.5, "dagger2": 2.5}, 10, config
    )
    assert not refit.source_weight_audit(
        {"clean": 5.0, "dagger1": 2.0, "dagger2": 3.0}, 10, config
    )


def test_variable_dataset_loader_supports_external_recovery_report() -> None:
    signature = inspect.signature(VariableGateBCDataset)
    assert "report_path" in signature.parameters
    assert refit.DAGGER2_RECOVERY_REPORT_SHA256 == (
        "38358a5813b7d6811d186006bc44a2e4470e51ad5a3d9c462bef4516b2b069c9"
    )
    assert refit.DAGGER2_ADMISSION_SHA256 == (
        "5c9d0e4bc954f45dd2f2b7609344f2966b43dd15e56f2cc7bf5adaea6f76d825"
    )


def test_vg014_frozen_inputs_verify() -> None:
    refit.verify_inputs()


def test_vg014_runner_is_source_locked_resumable_and_offline() -> None:
    text = refit.RUNNER.read_text()
    assert os.access(refit.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git status --porcelain --untracked-files=no" in text
    assert refit.TAG in text
    assert "tests/test_train_vq2_variable_gate_three_source_refit.py" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
