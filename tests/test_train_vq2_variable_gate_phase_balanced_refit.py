from __future__ import annotations

import inspect
import os

import pytest
import torch

from pufferlib.vq2_recurrent import ACTION_SIZE
import scripts.train_vq2_variable_gate_phase_balanced_refit as vg042
import scripts.train_vq2_variable_gate_seven_source_refit as core


def test_vg042_fixed_outer_and_phase_weights() -> None:
    config = vg042.PhaseBalancedConfig()
    assert config.seed == 429156
    assert config.epochs == 4
    assert config.learning_rate == pytest.approx(2e-6)
    assert config.sequence_chunk == 256
    assert config.transition_window_exposure == 4
    assert vg042.SOURCE_WEIGHTS == {
        "clean": 0.25,
        "dagger1": 0.03,
        "dagger2": 0.03,
        "dagger3": 0.06,
        "dagger4": 0.08,
        "dagger5": 0.08,
        "dagger6": 0.12,
        "dagger7": 0.15,
        "dagger8": 0.20,
    }
    assert sum(vg042.SOURCE_WEIGHTS.values()) == pytest.approx(1.0)
    assert vg042.PHASE_ROW_WEIGHTS[:6] == (
        0.25,
        0.50,
        1.00,
        6.00,
        32.00,
        32.00,
    )
    assert len(vg042.PHASE_ROW_WEIGHTS) == 17


def test_vg042_row_weights_decode_only_public_phase() -> None:
    observation = torch.zeros((1, 6, vg042.PHASE_LEGAL_OBS_SIZE))
    observation[0, :, -1] = torch.arange(6) / 16.0
    valid = torch.ones((1, 6), dtype=torch.bool)
    weights = vg042.phase_row_weights(observation, valid)
    assert torch.equal(
        weights,
        torch.tensor([[0.25, 0.5, 1.0, 6.0, 32.0, 32.0]]),
    )
    observation[0, 2, -1] += 0.01
    with pytest.raises(RuntimeError, match="non-/16"):
        vg042.phase_row_weights(observation, valid)


def test_vg042_phase_evaluator_imports_agent_batches_from_defining_module() -> None:
    batches = list(vg042._agent_batches(torch.arange(5).numpy(), 2))
    assert [batch.tolist() for batch in batches] == [[0, 1], [2, 3], [4]]
    source = inspect.getsource(vg042.evaluate_phase_balanced)
    assert "for batch_agents in _agent_batches(" in source
    assert "core._agent_batches" not in source


def test_refit_core_applies_row_weights_inside_one_source_only() -> None:
    config = core.SevenSourceConfig(smoothness_weight=0.0)
    names = tuple(core.SOURCE_WEIGHTS)
    predictions = {
        name: torch.zeros((1, 2, ACTION_SIZE)) for name in names
    }
    targets = {name: torch.zeros_like(predictions[name]) for name in names}
    targets["clean"][0, 0] = 1.0
    targets["clean"][0, 1] = 3.0
    valid = {
        name: torch.ones((1, 2), dtype=torch.bool) for name in names
    }
    kwargs = {
        "weights": torch.tensor(config.action_weights),
        "previous_predictions": {name: None for name in names},
        "previous_valid": {name: None for name in names},
        "config": config,
    }
    default, _ = core.seven_source_loss(
        predictions, targets, valid, **kwargs
    )
    weighted, _ = core.seven_source_loss(
        predictions,
        targets,
        valid,
        row_weights={"clean": torch.tensor([[1.0, 3.0]])},
        **kwargs,
    )
    assert default.item() == pytest.approx(1.25)
    assert weighted.item() == pytest.approx(1.75)


def test_vg042_admission_requires_phase4_and_balanced_improvement() -> None:
    config = vg042.PhaseBalancedConfig()
    baseline = {
        "source_balanced_weighted_mse": 0.10,
        "dagger8": {
            "weighted_mse": 0.10,
            "per_phase_weighted_mse": {"3": 0.08, "4": 0.12},
        },
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
        "dagger8": {
            "weighted_mse": 0.07,
            "unweighted_weighted_mse": 0.06,
            "per_phase_weighted_mse": {"3": 0.08, "4": 0.09},
        },
    }
    assert vg042.numerical_admission_predicate(
        best_epoch=1,
        best_score=0.06,
        baseline_validation=baseline,
        selected_validation=selected,
        source_balance_audit=True,
        config=config,
    )
    selected["dagger8"]["per_phase_weighted_mse"]["4"] = 0.12
    assert not vg042.numerical_admission_predicate(
        best_epoch=1,
        best_score=0.06,
        baseline_validation=baseline,
        selected_validation=selected,
        source_balance_audit=True,
        config=config,
    )


def test_vg042_configures_causal_phase_balanced_core_hooks() -> None:
    vg042.configure_core()
    assert core.ROW_OBJECTIVE_WEIGHTERS == {
        "dagger8": vg042.phase_row_weights
    }
    assert core.DATASET_EVALUATORS == {
        "dagger8": vg042.evaluate_phase_balanced
    }
    assert core.EXTRA_TRAINING_IDENTITY["causal_sequences_preserved"]
    source = inspect.getsource(core.train)
    assert "ROW_OBJECTIVE_WEIGHTERS" in source
    assert "row_weights=row_weights" in source


def test_vg042_inputs_and_runner_remain_offline() -> None:
    required = (
        vg042.vg040.PARENT_CHECKPOINT,
        vg042.vg040.PARENT_REPORT,
        vg042.vg040.VG039_DATASET / "report.json",
        vg042.vg040.VG039_DATASET / "metadata.json",
    )
    if all(path.is_file() for path in required):
        vg042.verify_inputs()
    text = vg042.RUNNER.read_text()
    assert os.access(vg042.RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "OMP_NUM_THREADS=4 MKL_NUM_THREADS=1" in text
    assert "tests/test_train_vq2_variable_gate_phase_balanced_refit.py" in text
    assert "--resume" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
