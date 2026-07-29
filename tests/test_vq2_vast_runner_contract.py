from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_vq2_vg002_vast.sh"
SYNC = ROOT / "scripts/sync_vq2_vg002_vast_evidence.sh"


def test_vast_runner_is_source_locked_resumable_and_flightsim_free() -> None:
    text = RUNNER.read_text()
    assert os.access(RUNNER, os.X_OK)
    assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
    assert 'git rev-parse HEAD' in text
    assert 'git status --porcelain --untracked-files=no' in text
    assert 'bash build.sh drone_race_vision --float' in text
    assert 'assert _C.precision_bytes == 4' in text
    assert 'scripts/eval_vq2_variable_gate_oracle.py' in text
    assert '--counts 5 8 11 12' in text
    assert '--agents 512' in text
    assert '--episodes 512' in text
    assert '--seed 429020' in text
    assert '--resume' in text
    assert text.index('state.json') < text.index('apt-get update')
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text


def test_vast_sync_copies_only_offline_vg002_evidence() -> None:
    text = SYNC.read_text()
    assert os.access(SYNC, os.X_OK)
    assert "C.$VQ2_INSTANCE_ID:$VQ2_REMOTE_OUTPUT/" in text
    assert "vq2_vg002_runner.log" in text
    assert "logs/remote_sync" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
