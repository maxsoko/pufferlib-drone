from __future__ import annotations

import importlib
import os

import scripts.eval_vq2_vg058_residual_scale_bracket as vg058


def test_vg058_uses_fresh_offsets_and_residual_scales() -> None:
    assert vg058.OFFSETS == (96, 104, 112, 120)
    assert vg058.AGENTS == vg058.EPISODES == 32
    assert vg058.ALPHAS == (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)


def test_vg058_requires_gate5_and_early_safety() -> None:
    parent = {"all_hard_transport_pass": True, "successes": 0, "crashes": 5,
              "mean_gates_passed": 2.0,
              "gate_reach": {"1": 128, "2": 115, "3": 20, "4": 1, "5": 0, "6": 0}}
    candidate = {**parent, "gate_reach": {"1": 128, "2": 116, "3": 21,
                                           "4": 2, "5": 1, "6": 0}}
    assert vg058.qualifies(parent, candidate)
    candidate["gate_reach"]["5"] = 0
    assert not vg058.qualifies(parent, candidate)


def test_vg058_endpoints_and_runner_are_source_locked() -> None:
    required = (vg058.PARENT_CHECKPOINT, vg058.UPDATE_CHECKPOINT, vg058.UPDATE_ADMISSION)
    if all(path.is_file() for path in required):
        try:
            vg058.configure(); vg058.verify_inputs()
            parent, update = vg058.load_endpoints()
            assert parent["model"]["class"] == "VQ2PhaseResidualActor"
            assert parent["model_state"]["phase_action_residual.weight"].count_nonzero() == 0
            assert update["model_state"]["phase_action_residual.weight"].count_nonzero() > 0
        finally:
            importlib.reload(vg058.bracket)
    if vg058.RUNNER.is_file():
        text = vg058.RUNNER.read_text()
        assert os.access(vg058.RUNNER, os.X_OK)
        assert "--resume" in text
        for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
            assert forbidden not in text
