from __future__ import annotations

import os
import subprocess
import sys

import scripts.eval_vq2_vg049_offset16_micro_fraction_bracket as vg049


def test_vg049_fixed_micro_fraction_contract() -> None:
    assert vg049.SEED == 429163
    assert vg049.EPISODE_OFFSET == 16
    assert vg049.ALPHAS == (
        0.0,
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.0125,
        0.015,
        0.02,
        0.025,
    )


def test_vg049_qualification_accepts_safe_gate4_gain() -> None:
    parent = {
        "hard_transport_pass": True,
        "gate_reach": {"1": 64, "2": 56, "3": 10, "4": 0},
        "crashes": 8,
        "successes": 0,
    }
    candidate = {
        "hard_transport_pass": True,
        "gate_reach": {"1": 64, "2": 56, "3": 9, "4": 1},
        "crashes": 8,
        "successes": 0,
    }
    assert vg049.qualifies(parent, candidate)
    candidate["crashes"] = 9
    assert not vg049.qualifies(parent, candidate)


def test_vg049_inputs_bind_vg048_rejection() -> None:
    vg049.verify_inputs()


def test_vg049_wrapper_and_runner_execute_directly_offline() -> None:
    result = subprocess.run(
        [sys.executable, str(vg049.Path(vg049.__file__)), "--help"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    text = vg049.RUNNER.read_text()
    assert os.access(vg049.RUNNER, os.X_OK)
    assert "eval_vq2_vg049_offset16_micro_fraction_bracket.py" in text
    assert "--resume" in text
    for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
        assert forbidden not in text
