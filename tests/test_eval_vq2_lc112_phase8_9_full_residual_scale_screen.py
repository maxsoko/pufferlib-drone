from __future__ import annotations

import torch

import scripts.eval_vq2_lc112_phase8_9_full_residual_scale_screen as lc112


def test_lc112_source_lock_and_scale_contract() -> None:
    lc112.verify_inputs()
    assert lc112.ALPHAS == (0.0, 0.01, 0.03, 0.10)
    assert lc112.TARGET_RAW_INDEX == 10
    assert lc112.pairwise.GROUP_SIZE == 128
    assert lc112.pairwise.PAIR_SIZE == 256


def test_lc112_endpoint_interpolation_exactness() -> None:
    parent = lc112.verify_inputs()["model_state"]
    baseline = lc112.candidate_state_for_index(parent, 0)
    endpoint = lc112.candidate_state_for_index(parent, 3)
    assert all(torch.equal(baseline[name], value) for name, value in parent.items())
    assert not torch.equal(
        endpoint["indexed_phase_residual_input"][8],
        parent["indexed_phase_residual_input"][8],
    )
    for phase in range(24):
        if phase in lc112.training.PHASES:
            continue
        assert torch.equal(
            endpoint["indexed_phase_residual_input"][phase],
            parent["indexed_phase_residual_input"][phase],
        )
