from __future__ import annotations

import importlib
import os

import torch

import scripts.eval_vq2_vg059_phase4_indexed_residual_bracket as vg059


def test_vg059_contract_is_phase4_only_and_fresh() -> None:
    assert vg059.TARGET_PHASE_INDEX == 4
    assert vg059.SHARED_TO_INDEXED_SCALE == 0.25
    assert vg059.OFFSETS == (128, 136, 144, 152)
    assert vg059.AGENTS == vg059.EPISODES == 64
    assert vg059.ALPHAS[-1] == 4.0


def test_vg059_requires_exact_pre_phase4_reach_and_new_gate5() -> None:
    parent = {"all_hard_transport_pass": True, "successes": 0, "crashes": 3,
              "mean_gates_passed": 2.1,
              "gate_reach": {"1": 255, "2": 235, "3": 45, "4": 3, "5": 0, "6": 0}}
    candidate = {**parent, "gate_reach": {**parent["gate_reach"], "5": 1}}
    assert vg059.qualifies(parent, candidate)
    candidate["gate_reach"]["3"] = 44
    assert not vg059.qualifies(parent, candidate)


def test_vg059_mapping_is_exact_before_phase4() -> None:
    required = (vg059.PARENT_CHECKPOINT, vg059.UPDATE_CHECKPOINT, vg059.REJECTION)
    if all(path.is_file() for path in required):
        try:
            vg059.configure(); vg059.verify_inputs()
            parent, update = vg059.load_endpoints()
            p = parent["model_state"]["indexed_phase_action_residual"]
            u = update["model_state"]["indexed_phase_action_residual"]
            assert torch.count_nonzero(p) == 0
            assert torch.count_nonzero(u[:4]) == 0
            assert torch.count_nonzero(u[4]) > 0
            assert torch.count_nonzero(u[5:]) == 0
        finally:
            importlib.reload(vg059.bracket.interpolation)
            importlib.reload(vg059.bracket)
    if vg059.RUNNER.is_file():
        text = vg059.RUNNER.read_text()
        assert os.access(vg059.RUNNER, os.X_OK)
        assert "--resume" in text
        for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
            assert forbidden not in text
