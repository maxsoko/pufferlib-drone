from __future__ import annotations

import torch

import scripts.eval_vq2_lc089_phase7_parity_bias_confirmation as lc089


def test_lc089_confirmation_contract() -> None:
    assert lc089.GROUP_SIZE == 128
    assert lc089.GROUP_SIZE * len(lc089.BIASES) == 256
    assert lc089.TARGET_PHASE == 7
    assert lc089.TARGET_RAW_INDEX == 8
    assert lc089.MINIMUM_PASS_GAIN == 2


def test_lc089_source_lock_and_exact_half_bias() -> None:
    parent = lc089.verify_inputs()
    state = parent["model_state"]
    candidate = lc089.candidate_state_for_index(state, 1)
    delta = (
        candidate["indexed_phase_residual_output_bias"][lc089.TARGET_PHASE]
        - state["indexed_phase_residual_output_bias"][lc089.TARGET_PHASE]
    )
    assert torch.allclose(
        delta, torch.tensor(lc089.SELECTED_BIAS), atol=1e-8, rtol=0.0
    )
