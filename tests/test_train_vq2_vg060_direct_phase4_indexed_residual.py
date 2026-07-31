from __future__ import annotations

import importlib
import os

import scripts.train_vq2_vg060_direct_phase4_indexed_residual as vg060


def test_vg060_targets_only_phase4() -> None:
    assert vg060.TARGET_PHASE_INDEX == 4
    assert vg060.PHASE_ROW_WEIGHTS == (0.0, 0.0, 0.0, 0.0, 1.0, *([0.0] * 12))
    assert vg060.CONFIG.epochs == 12
    assert vg060.CONFIG.sequence_chunk == 256


def test_vg060_admission_requires_exact_early_phases() -> None:
    baseline = {"phase_weighted_mse": {str(i): 1.0 for i in range(5)}}
    selected = {"phase_counts": {"4": 1000},
                "phase_weighted_mse": {**baseline["phase_weighted_mse"], "4": 0.8}}
    assert vg060.numerically_admitted(
        baseline, selected, base_parameters_exact=True, residual_norm=2.0
    )
    selected["phase_weighted_mse"]["3"] = 0.999
    assert not vg060.numerically_admitted(
        baseline, selected, base_parameters_exact=True, residual_norm=2.0
    )


def test_vg060_inputs_and_runner_are_source_locked() -> None:
    required = (vg060.core.PARENT_CHECKPOINT, vg060.core.DATASET / "report.json", vg060.REJECTION)
    if all(path.is_file() for path in required):
        try:
            vg060.configure(); vg060.verify_inputs()
        finally:
            importlib.reload(vg060.core)
    if vg060.RUNNER.is_file():
        text = vg060.RUNNER.read_text()
        assert os.access(vg060.RUNNER, os.X_OK)
        assert "--resume" in text
        for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
            assert forbidden not in text
