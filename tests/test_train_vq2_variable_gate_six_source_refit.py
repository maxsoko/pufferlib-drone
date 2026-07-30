from __future__ import annotations

import inspect
import os

import pytest
import torch

from pufferlib.vq2_recurrent import ACTION_SIZE
from scripts.train_vq2_recurrent_bc import weighted_action_mse
import scripts.train_vq2_variable_gate_six_source_refit as refit


def test_six_source_loss_uses_fixed_weights_despite_record_counts() -> None:
    config = refit.SixSourceConfig(smoothness_weight=0.0)
    weights = torch.tensor(config.action_weights)
    lengths = {
        "clean": 1,
        "dagger1": 3,
        "dagger2": 5,
        "dagger3": 7,
        "dagger4": 9,
        "dagger5": 11,
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
    loss, components = refit.six_source_loss(
        predictions,
        targets,
        valid,
        weights=weights,
        previous_predictions={name: None for name in lengths},
        previous_valid={name: None for name in lengths},
        config=config,
    )
    expected = torch.zeros(())
    for name, source_weight in refit.SOURCE_WEIGHTS.items():
        source_loss, _ = weighted_action_mse(
            predictions[name], targets[name], valid[name], weights
        )
        expected = expected + source_weight * source_loss
        assert torch.equal(components[f"{name}_total"], source_loss)
    assert torch.equal(loss, expected)


def test_vg025_configuration_and_source_weight_audit_are_exact() -> None:
    config = refit.SixSourceConfig()
    assert config.seed == 429110
    assert config.epochs == 6
    assert config.sequence_chunk == 256
    assert config.transition_window_exposure == 4
    assert config.learning_rate == pytest.approx(5e-6)
    assert refit.objective_weights(config) == {
        "clean": 0.30,
        "dagger1": 0.05,
        "dagger2": 0.05,
        "dagger3": 0.10,
        "dagger4": 0.15,
        "dagger5": 0.35,
    }
    expected_sums = {name: 10 * value for name, value in refit.SOURCE_WEIGHTS.items()}
    assert refit.source_weight_audit(expected_sums, 10, config)
    expected_sums["dagger5"] += 1e-5
    assert not refit.source_weight_audit(expected_sums, 10, config)


def test_vg025_frozen_inputs_and_all_dataset_loaders_verify() -> None:
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
        "dagger5": refit.VariableGateBCDataset(
            refit.DAGGER5_DATASET,
            expected_report_sha256=refit.DAGGER5_REPORT_SHA256,
            expected_metadata_sha256=refit.DAGGER5_METADATA_SHA256,
        ),
    }
    assert {name: dataset.agents for name, dataset in datasets.items()} == {
        "clean": 256,
        "dagger1": 512,
        "dagger2": 512,
        "dagger3": 512,
        "dagger4": 512,
        "dagger5": 512,
    }
    assert datasets["dagger5"].phase_audit == {
            "increments": 519,
        "encoding_max_error": 0.0,
        "invalid_padding_rows_excluded": 11_937,
    }


def test_vg025_parent_loads_and_preserves_actor_contract() -> None:
    config = refit.SixSourceConfig()
    model = refit.VQ2PhaseRecurrentActor(
        hidden_size=config.hidden_size, initial_std=config.initial_std
    )
    payload = refit._load_parent(model)
    assert payload["tag"] == "vq2_vg022_five_source_device_decode_continuation_001"
    assert payload["best_epoch"] == 9
    assert payload["numerically_admitted"] is True


def test_vg025_source_surface_binds_every_source_and_admission() -> None:
    paths = set(refit.source_paths())
    for required in (
        refit.PREREGISTRATION,
        refit.RUNNER,
        refit.DAGGER1_ADMISSION,
        refit.DAGGER2_ADMISSION,
        refit.DAGGER3_ADMISSION,
        refit.DAGGER4_ADMISSION,
        refit.DAGGER5_ADMISSION,
        refit.PARENT_ADMISSION,
        refit.DAGGER5_DATASET / "report.json",
        refit.DAGGER5_DATASET / "metadata.json",
        refit.PARENT_CHECKPOINT,
        refit.PARENT_REPORT,
    ):
        assert required.resolve() in paths


def test_vg025_runner_is_source_locked_resumable_and_offline() -> None:
    text = refit.RUNNER.read_text()
    assert os.access(refit.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git status --porcelain --untracked-files=no" in text
    assert refit.TAG in text
    assert "tests/test_train_vq2_variable_gate_six_source_refit.py" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text


def test_vg025_resume_restores_cpu_rng_payload_before_cuda_state() -> None:
    source = inspect.getsource(refit.train)
    assert 'torch.load(state_path, map_location="cpu", weights_only=False)' in source
    assert '"flight_sim_packets_sent": 0' in source
    assert '"teacher_blend": 0.0' in source
