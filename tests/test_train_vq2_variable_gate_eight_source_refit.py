from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os

import pytest

import scripts.train_vq2_variable_gate_eight_source_refit as refit
import scripts.train_vq2_variable_gate_seven_source_refit as core


@pytest.fixture(autouse=True)
def restore_core_extension_surface():
    names = (
        "TAG",
        "REPORT_SCHEMA",
        "STATE_SCHEMA",
        "SEED",
        "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256",
        "PARENT_REPORT",
        "PARENT_REPORT_SHA256",
        "PARENT_ADMISSION",
        "PARENT_ADMISSION_SHA256",
        "PREREGISTRATION",
        "RUNNER",
        "DEFAULT_OUTPUT",
        "SOURCE_WEIGHTS",
        "EXTRA_DATASET_SPECS",
        "EXTRA_OBJECTIVE_WEIGHT_ATTRIBUTES",
        "NUMERICAL_ADMISSION_PREDICATE",
        "verify_inputs",
        "_load_parent",
        "source_paths",
        "current_source_identity",
    )
    saved = {name: getattr(core, name) for name in names}
    yield
    for name, value in saved.items():
        setattr(core, name, value)


def _validation(value: float = 0.01) -> dict[str, dict[str, float]]:
    return {
        name: {"weighted_mse": value}
        for name in refit.SOURCE_WEIGHTS
    }


def test_vg033_configuration_uses_all_eight_explicit_sources() -> None:
    config = refit.EightSourceConfig()
    assert config.seed == 429132
    assert config.epochs == 6
    assert config.sequence_chunk == 256
    assert config.transition_window_exposure == 4
    assert sum(refit.SOURCE_WEIGHTS.values()) == pytest.approx(1.0)
    assert refit.SOURCE_WEIGHTS["clean"] == pytest.approx(0.25)
    assert refit.SOURCE_WEIGHTS["dagger7"] == pytest.approx(0.35)
    assert asdict(config)["dagger7_validation_agents"] == 64


def test_vg033_numerical_admission_requires_new_source_improvement_and_caps() -> None:
    config = refit.EightSourceConfig()
    baseline = _validation(0.20)
    baseline["source_balanced_weighted_mse"] = 0.20
    selected = _validation(0.01)
    assert refit.numerical_admission_predicate(
        best_epoch=1,
        best_score=0.05,
        baseline_validation=baseline,
        selected_validation=selected,
        source_balance_audit=True,
        config=config,
    )

    selected["dagger7"]["weighted_mse"] = 0.21
    assert not refit.numerical_admission_predicate(
        best_epoch=1,
        best_score=0.05,
        baseline_validation=baseline,
        selected_validation=selected,
        source_balance_audit=True,
        config=config,
    )


def test_vg033_configures_generic_core_without_changing_actor_abi() -> None:
    refit.configure_core()
    config = refit.EightSourceConfig()
    assert core.TAG == refit.TAG
    assert core.SOURCE_WEIGHTS == refit.SOURCE_WEIGHTS
    assert core.objective_weights(config) == refit.SOURCE_WEIGHTS
    assert set(core.EXTRA_DATASET_SPECS) == {"dagger7"}
    assert core.EXTRA_DATASET_SPECS["dagger7"]["dataset"] == refit.VG032_DATASET
    assert core.NUMERICAL_ADMISSION_PREDICATE is (
        refit.numerical_admission_predicate
    )


def test_vg033_source_surface_includes_new_dataset_and_wrapper() -> None:
    paths = set(refit.source_paths())
    assert refit.VG032_DATASET / "report.json" in paths
    assert refit.VG032_DATASET / "metadata.json" in paths
    assert refit.VG032_ADMISSION in paths
    assert Path(refit.__file__).resolve() in paths


def test_vg033_frozen_new_inputs_match_admitted_vg032() -> None:
    assert refit.VG032_REPORT_SHA256 == (
        "f725afd984ae5a02999bd01560b24d4728955cc8e4a3c1cad972591ed9556d41"
    )
    assert refit.VG032_METADATA_SHA256 == (
        "b4d43848fe217ee490a57ff19f0eeff5752a58633c7181fde3472700b41a9701"
    )
    assert refit.VG032_ADMISSION_SHA256 == (
        "7887a69aba40a2c364336a516ef0ef3e45377118af149c4a27bc3376ee88e0be"
    )


def test_vg033_runner_is_source_locked_resumable_and_offline() -> None:
    text = refit.RUNNER.read_text()
    assert os.access(refit.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git status --porcelain --untracked-files=no" in text
    assert refit.TAG in text
    assert "tests/test_train_vq2_variable_gate_eight_source_refit.py" in text
    assert "bash build.sh drone_race_vision --float" in text
    assert "--resume" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
