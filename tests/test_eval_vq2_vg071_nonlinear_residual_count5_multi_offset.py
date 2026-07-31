from __future__ import annotations

import torch

from scripts.eval_vq2_vg071_nonlinear_residual_count5_multi_offset import (
    AGENTS,
    ALPHAS,
    MAX_STEPS,
    NUM_GATES,
    OFFSETS,
    load_endpoints,
    qualifies,
)


def test_paired_endpoints_are_exact_zero_and_selected() -> None:
    parent, selected = load_endpoints()
    p = parent["model_state"]
    s = selected["model_state"]
    for name in s:
        if name not in {
            "indexed_phase_residual_output",
            "indexed_phase_residual_output_bias",
        }:
            torch.testing.assert_close(p[name], s[name], rtol=0, atol=0)
    assert torch.count_nonzero(p["indexed_phase_residual_output"]) == 0
    assert torch.count_nonzero(p["indexed_phase_residual_output_bias"]) == 0
    assert torch.count_nonzero(s["indexed_phase_residual_output"]) > 0


def test_confirmation_qualification_is_finish_and_crash_strict() -> None:
    parent = {
        "all_hard_transport_pass": True, "successes": 20, "crashes": 10,
        "gate_reach": {"1": 128},
    }
    candidate = {
        "all_hard_transport_pass": True, "successes": 21, "crashes": 10,
        "gate_reach": {"1": 128},
    }
    assert qualifies(parent, candidate)
    candidate["successes"] = 20
    assert not qualifies(parent, candidate)
    candidate["successes"] = 21; candidate["crashes"] = 11
    assert not qualifies(parent, candidate)


def test_fixed_multi_offset_contract() -> None:
    assert ALPHAS == (0.0, 1.0)
    assert OFFSETS == (216, 224, 232, 240)
    assert AGENTS == 32
    assert NUM_GATES == 5
    assert MAX_STEPS == 23040
