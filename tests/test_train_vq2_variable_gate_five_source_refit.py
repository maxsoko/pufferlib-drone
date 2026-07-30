from __future__ import annotations

import inspect
import json
import os
from dataclasses import replace

import pytest
import torch

from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.train_vq2_recurrent_bc import weighted_action_mse
import scripts.train_vq2_variable_gate_five_source_refit as refit


def test_five_source_loss_uses_fixed_weights_despite_record_counts() -> None:
    config = replace(refit.FiveSourceConfig(), smoothness_weight=0.0)
    weights = torch.tensor(config.action_weights)
    lengths = {
        "clean": 1,
        "dagger1": 5,
        "dagger2": 9,
        "dagger3": 13,
        "dagger4": 17,
    }
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
    loss, components = refit.five_source_loss(
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


def test_vg020_configuration_and_source_weight_audit_are_exact() -> None:
    config = refit.FiveSourceConfig()
    assert config.sequence_chunk == 256
    assert config.transition_window_exposure == 4
    assert config.learning_rate == pytest.approx(1e-5)
    assert refit.objective_weights(config) == {
        "clean": 0.35,
        "dagger1": 0.10,
        "dagger2": 0.10,
        "dagger3": 0.15,
        "dagger4": 0.30,
    }
    assert refit.source_weight_audit(
        {
            "clean": 3.5,
            "dagger1": 1.0,
            "dagger2": 1.0,
            "dagger3": 1.5,
            "dagger4": 3.0,
        },
        10,
        config,
    )
    assert not refit.source_weight_audit(
        {
            "clean": 3.0,
            "dagger1": 1.0,
            "dagger2": 1.0,
            "dagger3": 2.0,
            "dagger4": 3.0,
        },
        10,
        config,
    )


def test_vg020_frozen_inputs_and_all_dataset_loaders_verify() -> None:
    refit.verify_inputs()
    datasets = {
        "clean": refit.VariableGateBCDataset(
            refit.CLEAN_DATASET,
            expected_report_sha256=refit.CLEAN_REPORT_SHA256,
            expected_metadata_sha256=refit.CLEAN_METADATA_SHA256,
        ),
        "dagger1": refit.VariableGateBCDataset(
            refit.DAGGER1_DATASET,
            expected_report_sha256=refit.DAGGER1_REPORT_SHA256,
            expected_metadata_sha256=refit.DAGGER1_METADATA_SHA256,
        ),
        "dagger2": refit.VariableGateBCDataset(
            refit.DAGGER2_DATASET,
            report_path=refit.DAGGER2_RECOVERY_REPORT,
            expected_report_sha256=refit.DAGGER2_RECOVERY_REPORT_SHA256,
            expected_metadata_sha256=refit.DAGGER2_METADATA_SHA256,
        ),
        "dagger3": refit.VariableGateBCDataset(
            refit.DAGGER3_DATASET,
            expected_report_sha256=refit.DAGGER3_REPORT_SHA256,
            expected_metadata_sha256=refit.DAGGER3_METADATA_SHA256,
        ),
        "dagger4": refit.VariableGateBCDataset(
            refit.DAGGER4_DATASET,
            expected_report_sha256=refit.DAGGER4_REPORT_SHA256,
            expected_metadata_sha256=refit.DAGGER4_METADATA_SHA256,
        ),
    }
    assert {name: dataset.agents for name, dataset in datasets.items()} == {
        "clean": 256,
        "dagger1": 512,
        "dagger2": 512,
        "dagger3": 512,
        "dagger4": 512,
    }
    assert datasets["dagger4"].phase_audit == {
        "increments": 521,
        "encoding_max_error": 0.0,
        "invalid_padding_rows_excluded": 48_835,
    }


def test_vg020_parent_loads_and_preserves_actor_contract() -> None:
    config = refit.FiveSourceConfig()
    model = refit.VQ2PhaseRecurrentActor(
        hidden_size=config.hidden_size, initial_std=config.initial_std
    )
    payload = refit._load_parent(model)
    assert payload["tag"] == "vq2_vg017_variable_gate_four_source_refit_001"
    assert payload["best_epoch"] == 11
    assert payload["numerically_admitted"] is True


def test_vg020_runner_is_source_locked_resumable_and_offline() -> None:
    text = refit.RUNNER.read_text()
    assert os.access(refit.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git status --porcelain --untracked-files=no" in text
    assert refit.TAG in text
    assert "tests/test_train_vq2_variable_gate_five_source_refit.py" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text


def test_vg022_migration_and_parity_gate_are_frozen_before_epoch_four() -> None:
    assert refit.TAG == "vq2_vg022_five_source_device_decode_continuation_001"
    assert refit.MIGRATION_STATE_SHA256 == (
        "3d4694baaeac55184d3672675b4c159ee339adbff45c4b46246a58118ecbb14b"
    )
    preregistration = refit.PREREGISTRATION.read_text()
    runner = refit.RUNNER.read_text()
    checker = refit.PARITY_CHECKER.read_text()
    assert refit.MIGRATION_STATE_SHA256 in preregistration
    assert refit.MIGRATION_STATE_SHA256 in runner
    assert "`3`" in preregistration
    assert "31,742" in preregistration
    assert runner.rindex("check_vq2_vg022_device_decode_parity.py") < runner.rindex(
        "run_training"
    )
    assert "forward_source_groups" not in checker
    assert "forward_source_groups" not in refit.train.__code__.co_names
    assert '"optimizer_steps": 0' in checker
    assert '"state_writes": 0' in checker
    for forbidden in ("apt-get", "pip install", "FlightSim", "14550", "5600"):
        assert forbidden not in runner
        assert forbidden not in checker


def test_vg022_restores_rng_payload_from_cpu_after_bootstrap_three() -> None:
    source = inspect.getsource(refit.train)
    assert 'torch.load(state_path, map_location="cpu", weights_only=False)' in source
    assert (
        'torch.load(migration_state, map_location="cpu", weights_only=False)'
        in source
    )
    failure = json.loads(refit.VG022_BOOTSTRAP3_FAILURE.read_text())
    assert failure["bootstrap"] == 3
    assert failure["parity"]["admitted"] is True
    assert failure["parity"]["optimizer_steps"] == 0
    assert failure["vg022_state_created"] is False
