from __future__ import annotations

import torch

import scripts.eval_vq2_lc086_phase6_half_bias_confirmation as lc086


def test_lc086_confirmation_contract() -> None:
    assert lc086.GROUP_SIZE == 128
    assert lc086.GROUP_SIZE * len(lc086.BIASES) == 256
    assert lc086.TARGET_PHASE == 6
    assert lc086.TARGET_RAW_INDEX == 7
    assert lc086.MINIMUM_PASS_GAIN == 2


def test_lc086_source_lock_and_exact_half_bias() -> None:
    parent = lc086.verify_inputs()
    state = parent["model_state"]
    candidate = lc086.candidate_state_for_index(state, 1)
    delta = (
        candidate["indexed_phase_residual_output_bias"][lc086.TARGET_PHASE]
        - state["indexed_phase_residual_output_bias"][lc086.TARGET_PHASE]
    )
    assert torch.allclose(
        delta, torch.tensor(lc086.SELECTED_BIAS), atol=1e-8, rtol=0.0
    )
