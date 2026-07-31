from __future__ import annotations

import importlib
import os

import torch

import scripts.eval_vq2_vg061_direct_phase4_residual_bracket as vg061


def test_vg061_contract_is_fresh_and_bounded() -> None:
    assert vg061.OFFSETS == (160, 168, 176, 184)
    assert vg061.AGENTS == vg061.EPISODES == 64
    assert vg061.ALPHAS == (0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)


def test_vg061_endpoint_has_only_head4() -> None:
    required = (vg061.PARENT_CHECKPOINT, vg061.UPDATE_CHECKPOINT, vg061.UPDATE_ADMISSION)
    if all(path.is_file() for path in required):
        try:
            vg061.configure(); vg061.verify_inputs()
            parent, update = vg061.load_endpoints()
            p = parent["model_state"]["indexed_phase_action_residual"]
            u = update["model_state"]["indexed_phase_action_residual"]
            assert torch.count_nonzero(p) == 0
            assert torch.count_nonzero(u[:4]) == 0
            assert torch.count_nonzero(u[4]) > 0
            assert torch.count_nonzero(u[5:]) == 0
        finally:
            importlib.reload(vg061.bracket.interpolation)
            importlib.reload(vg061.bracket)
    if vg061.RUNNER.is_file():
        text = vg061.RUNNER.read_text()
        assert os.access(vg061.RUNNER, os.X_OK)
        assert "--resume" in text
        for forbidden in ("FlightSim", "14550", "5600", "COMMAND_LONG"):
            assert forbidden not in text
