from __future__ import annotations

import importlib
import os

import scripts.eval_vq2_vg056_count6_paired_confirmation as vg056


def summary(*, gate2: int, gate3: int, gate4: int, gate5: int, crashes: int) -> dict:
    return {
        "all_hard_transport_pass": True,
        "gate_reach": {
            "1": 256, "2": gate2, "3": gate3,
            "4": gate4, "5": gate5, "6": 0,
        },
        "successes": 0,
        "crashes": crashes,
        "mean_gates_passed": 2.1,
    }


def test_vg056_is_larger_and_independent() -> None:
    assert vg056.ALPHAS == (0.0, 1.0)
    assert vg056.OFFSETS == (64, 72, 80, 88)
    assert vg056.AGENTS == vg056.EPISODES == 64
    assert vg056.MAX_STEPS == 3072


def test_vg056_requires_gate5_and_early_safety_preservation() -> None:
    parent = summary(gate2=230, gate3=40, gate4=4, gate5=0, crashes=12)
    candidate = summary(gate2=231, gate3=41, gate4=5, gate5=1, crashes=12)
    assert vg056.qualifies(parent, candidate)
    candidate["gate_reach"]["5"] = 0
    assert not vg056.qualifies(parent, candidate)
    candidate["gate_reach"]["5"] = 1
    candidate["gate_reach"]["2"] = 229
    assert not vg056.qualifies(parent, candidate)


def test_vg056_inputs_and_runner_are_source_locked_offline() -> None:
    required = (vg056.UPDATE_CHECKPOINT, vg056.UPDATE_REPORT, vg056.UPDATE_ADMISSION)
    if all(path.is_file() for path in required):
        try:
            vg056.configure()
            vg056.bracket.verify_inputs()
            parent, update = vg056.bracket.load_endpoints()
            assert parent["model"] == update["model"]
        finally:
            importlib.reload(vg056.bracket)
    if vg056.RUNNER.is_file():
        text = vg056.RUNNER.read_text()
        assert os.access(vg056.RUNNER, os.X_OK)
        assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
        assert "--resume" in text
        for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
            assert forbidden not in text
