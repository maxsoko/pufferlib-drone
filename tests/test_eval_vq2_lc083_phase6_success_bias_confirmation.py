from __future__ import annotations

import torch

import scripts.eval_vq2_lc083_phase6_success_bias_confirmation as lc083


def test_lc083_confirmation_contract() -> None:
    assert lc083.GROUP_SIZE == 128
    assert lc083.GROUP_SIZE * len(lc083.BIASES) == 256
    assert lc083.TARGET_PHASE == 6
    assert lc083.TARGET_RAW_INDEX == 7
    assert lc083.MINIMUM_PASS_GAIN == 2
    assert lc083.MAX_STEPS == 12_000


def test_lc083_source_lock_and_exact_bias() -> None:
    parent = lc083.verify_inputs()
    state = parent["model_state"]
    candidate = lc083.candidate_state_for_index(state, 1)
    delta = (
        candidate["indexed_phase_residual_output_bias"][lc083.TARGET_PHASE]
        - state["indexed_phase_residual_output_bias"][lc083.TARGET_PHASE]
    )
    assert torch.allclose(
        delta, torch.tensor(lc083.SELECTED_BIAS), atol=1e-8, rtol=0.0
    )
