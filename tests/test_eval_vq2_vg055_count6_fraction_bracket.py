from __future__ import annotations

import os

import scripts.eval_vq2_vg055_count6_fraction_bracket as vg055


def test_vg055_uses_new_offsets_and_bounded_fractions() -> None:
    assert vg055.OFFSETS == (32, 40, 48, 56)
    assert vg055.AGENTS == vg055.EPISODES == 32
    assert vg055.MAX_STEPS == 3072
    assert vg055.ALPHAS == (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)


def test_vg055_qualification_requires_early_and_crash_preservation() -> None:
    parent = {
        "all_hard_transport_pass": True,
        "gate_reach": {"1": 128, "2": 120, "3": 24, "4": 2, "5": 0, "6": 0},
        "successes": 0,
        "crashes": 8,
        "mean_gates_passed": 2.1,
    }
    candidate = {
        **parent,
        "gate_reach": {"1": 128, "2": 120, "3": 25, "4": 2, "5": 0, "6": 0},
    }
    assert vg055.qualifies(parent, candidate)
    candidate["crashes"] = 9
    assert not vg055.qualifies(parent, candidate)
    candidate["crashes"] = 8
    candidate["gate_reach"]["2"] = 119
    assert not vg055.qualifies(parent, candidate)


def test_vg055_inputs_and_runner_are_source_locked_offline() -> None:
    required = (vg055.PARENT_CHECKPOINT, vg055.UPDATE_CHECKPOINT, vg055.REJECTION)
    if all(path.is_file() for path in required):
        vg055.verify_inputs()
        parent, update = vg055.load_endpoints()
        assert parent["model"] == update["model"]
    if vg055.RUNNER.is_file():
        text = vg055.RUNNER.read_text()
        assert os.access(vg055.RUNNER, os.X_OK)
        assert 'VQ2_EXPECTED_COMMIT="${VQ2_EXPECTED_COMMIT:?' in text
        assert "--resume" in text
        for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
            assert forbidden not in text
