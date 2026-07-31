from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import scripts.train_vq2_vg053_late_gate_trust_refit as vg053
from pufferlib.vq2_informed import MASK_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.train_vq2_variable_gate_recurrent_bc import (
    PHASE_TAIL_INDEX,
    audit_public_phase_layout,
    phase_increment_rows,
)


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


def test_vg053_allows_only_the_initial_late_phase_jump() -> None:
    tail = np.zeros(
        (4, 1, PHASE_LEGAL_OBS_SIZE - MASK_SIZE), dtype=np.float32
    )
    phase = np.asarray(
        [3.0 / 16.0, 3.0 / 16.0, 4.0 / 16.0, 4.0 / 16.0],
        dtype=np.float32,
    )
    tail[:, 0, PHASE_TAIL_INDEX] = phase
    valid = np.ones((4, 1), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="skipped public index"):
        audit_public_phase_layout(tail, valid, np.asarray([4]))
    audit = audit_public_phase_layout(
        tail,
        valid,
        np.asarray([4]),
        allow_initial_phase_jump=True,
    )
    assert audit["increments"] == 1
    assert audit["initial_phase_jumps"] == 1
    assert audit["maximum_initial_phase_index"] == 3
    transition = phase_increment_rows(
        phase[:, None], valid, np.asarray([phase[0]], dtype=np.float32)
    )
    assert transition[:, 0].tolist() == [False, False, True, False]


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
