from __future__ import annotations

import os

import pytest

import scripts.train_vq2_variable_gate_nine_source_refit as vg040


def test_vg040_fixed_weights_and_recurrent_contract() -> None:
    config = vg040.NineSourceConfig()
    assert config.seed == 429153
    assert config.epochs == 6
    assert config.sequence_chunk == 256
    assert config.transition_window_exposure == 4
    assert config.learning_rate == 5e-6
    assert vg040.SOURCE_WEIGHTS == {
        "clean": 0.22,
        "dagger1": 0.02,
        "dagger2": 0.02,
        "dagger3": 0.04,
        "dagger4": 0.05,
        "dagger5": 0.05,
        "dagger6": 0.08,
        "dagger7": 0.12,
        "dagger8": 0.40,
    }
    assert sum(vg040.SOURCE_WEIGHTS.values()) == pytest.approx(1.0)


def test_vg040_inputs_bind_parent_and_new_admitted_dataset() -> None:
    required = (
        vg040.PARENT_CHECKPOINT,
        vg040.PARENT_REPORT,
        vg040.VG039_DATASET / "report.json",
        vg040.VG039_DATASET / "metadata.json",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("retained remote evidence is not mirrored in this checkout")
    vg040.verify_inputs()


def test_vg040_numerical_admission_requires_new_source_improvement() -> None:
    config = vg040.NineSourceConfig()
    baseline = {
        "source_balanced_weighted_mse": 0.10,
        "dagger8": {"weighted_mse": 0.20},
    }
    selected = {
        "clean": {"weighted_mse": 0.01},
        "dagger1": {"weighted_mse": 0.01},
        "dagger2": {"weighted_mse": 0.02},
        "dagger3": {"weighted_mse": 0.02},
        "dagger4": {"weighted_mse": 0.02},
        "dagger5": {"weighted_mse": 0.02},
        "dagger6": {"weighted_mse": 0.04},
        "dagger7": {"weighted_mse": 0.05},
        "dagger8": {"weighted_mse": 0.08},
    }
    assert vg040.numerical_admission_predicate(
        best_epoch=1,
        best_score=0.05,
        baseline_validation=baseline,
        selected_validation=selected,
        source_balance_audit=True,
        config=config,
    )
    selected["dagger8"]["weighted_mse"] = 0.20
    assert not vg040.numerical_admission_predicate(
        best_epoch=1,
        best_score=0.05,
        baseline_validation=baseline,
        selected_validation=selected,
        source_balance_audit=True,
        config=config,
    )


def test_vg040_runner_is_source_locked_resumable_and_offline() -> None:
    text = vg040.RUNNER.read_text()
    assert os.access(vg040.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert vg040.TAG in text
    assert "OMP_NUM_THREADS=4 MKL_NUM_THREADS=1" in text
    assert "tests/test_train_vq2_variable_gate_nine_source_refit.py" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index(
        "test_drone_race_native_regressions"
    )
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
