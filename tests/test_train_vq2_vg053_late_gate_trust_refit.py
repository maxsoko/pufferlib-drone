from __future__ import annotations

import os
from pathlib import Path

import pytest

import scripts.train_vq2_vg053_late_gate_trust_refit as vg053


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_vq2_vg053_vast.sh"


def test_vg053_schedule_is_one_small_anchor_heavy_update() -> None:
    config = vg053.LateGateTrustConfig()
    assert config.epochs == 1
    assert config.learning_rate == 5e-7
    assert config.sequence_chunk == 256
    assert config.transition_window_exposure == 4
    assert vg053.SOURCE_WEIGHTS["dagger9"] == 0.10
    assert sum(vg053.SOURCE_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(
        vg053.SOURCE_WEIGHTS[name]
        for name in (
            "clean",
            "dagger1",
            "dagger2",
            "dagger3",
            "dagger4",
            "dagger5",
            "dagger6",
            "dagger7",
        )
    ) == pytest.approx(0.85)


def test_vg053_numerical_admission_requires_late_gate_improvement() -> None:
    config = vg053.LateGateTrustConfig()
    baseline = {
        "source_balanced_weighted_mse": 0.10,
        **{name: {"weighted_mse": 0.10} for name in vg053.SOURCE_WEIGHTS},
    }
    selected = {
        name: {"weighted_mse": 0.01} for name in vg053.SOURCE_WEIGHTS
    }
    assert vg053.numerical_admission_predicate(
        best_epoch=1,
        best_score=0.09,
        baseline_validation=baseline,
        selected_validation=selected,
        source_balance_audit=True,
        config=config,
    )
    selected["dagger9"]["weighted_mse"] = 0.10
    assert not vg053.numerical_admission_predicate(
        best_epoch=1,
        best_score=0.09,
        baseline_validation=baseline,
        selected_validation=selected,
        source_balance_audit=True,
        config=config,
    )


def test_vg053_source_evidence_verifies_when_mirrored() -> None:
    required = (
        vg053.VG052_DATASET / "report.json",
        vg053.VG052_DATASET / "metadata.json",
        vg053.VG052_ADMISSION,
        vg053.vg040.VG039_DATASET / "report.json",
        vg053.vg040.VG039_DATASET / "metadata.json",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("retained remote training evidence is not mirrored")
    vg053.verify_inputs()


def test_vg053_runner_is_resumable_source_locked_and_offline() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert "--resume" in text
    assert text.index('VQ2_STATE="') < text.index("python -m pytest")
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
