from __future__ import annotations

import os
import subprocess
import sys

import scripts.eval_vq2_vg051_count6_multi_offset_baseline as vg051
import scripts.eval_vq2_staged_count5_component as component


def test_component_supports_official_six_gate_proxy() -> None:
    assert 6 in component.ALLOWED_COUNTS


def test_vg051_fixed_multi_offset_contract() -> None:
    assert vg051.OFFSETS == (0, 8, 16, 24)
    assert vg051.AGENTS == vg051.EPISODES == 64
    assert vg051.THREADS == 4
    assert vg051.MAX_STEPS == 3072


def test_vg051_inputs_are_source_locked() -> None:
    vg051.verify_inputs()


def test_vg051_executes_directly_and_runner_is_offline() -> None:
    result = subprocess.run(
        [sys.executable, str(vg051.Path(vg051.__file__)), "--help"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    text = vg051.RUNNER.read_text()
    assert os.access(vg051.RUNNER, os.X_OK)
    assert "eval_vq2_vg051_count6_multi_offset_baseline.py" in text
    assert "--resume" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
