from __future__ import annotations

import os

import torch

import scripts.train_vq2_vg057_warmed_phase_residual as vg057


def test_vg057_contract_targets_only_warmed_phase_residual() -> None:
    config = vg057.TrainConfig()
    assert config.epochs == 6
    assert config.sequence_chunk == 256
    assert config.validation_agents == 64
    assert config.phase_row_weights[0] == 0.0
    assert config.phase_row_weights[4] == 32.0
    assert sum(p.numel() for p in torch.nn.Linear(256, 4, bias=False).parameters()) == 1024


def test_vg057_numerical_admission_requires_phase4_and_stability() -> None:
    baseline = {"phase_weighted_mse": {"2": 1.0, "3": 1.0, "4": 1.0}}
    selected = {"phase_counts": {"4": 1000},
                "phase_weighted_mse": {"2": 1.01, "3": 1.01, "4": 0.9}}
    assert vg057.numerically_admitted(
        baseline, selected, base_parameters_exact=True, residual_norm=1.0
    )
    selected["phase_weighted_mse"]["4"] = 1.0
    assert not vg057.numerically_admitted(
        baseline, selected, base_parameters_exact=True, residual_norm=1.0
    )


def test_vg057_inputs_and_runner_are_offline_source_locked() -> None:
    required = (vg057.PARENT_CHECKPOINT, vg057.DATASET / "report.json", vg057.REJECTION)
    if all(path.is_file() for path in required):
        vg057.verify_inputs()
    if vg057.RUNNER.is_file():
        text = vg057.RUNNER.read_text()
        assert os.access(vg057.RUNNER, os.X_OK)
        assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
        assert "--resume" in text
        for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
            assert forbidden not in text
