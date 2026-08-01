from __future__ import annotations

import torch

import scripts.eval_vq2_lc108_phase8_pairwise_scale_screen as lc108


def test_lc108_source_lock_and_pairwise_contract() -> None:
    lc108.configure()
    lc108.verify_inputs()
    assert lc108.TARGET_PHASE == 8
    assert lc108.TARGET_RAW_INDEX == 9
    assert lc108.ALPHAS == (0.0, 0.25, 0.50, 1.0)
    assert lc108.GROUP_SIZE == 128
    assert lc108.pairwise.PAIR_SIZE == 256
    assert lc108.milestone.TOTAL_AGENTS == 512


def test_lc108_exact_scale_uses_fitted_phase8_row() -> None:
    lc108.configure()
    parent = lc108.verify_inputs()
    fitted = torch.load(lc108.FIT_CHECKPOINT, map_location="cpu", weights_only=False)
    exact = lc108.pairwise.prior.candidate_state_for_index(
        parent["model_state"], lc108.ALPHAS.index(1.0)
    )
    assert torch.equal(
        exact["indexed_phase_residual_output"][8], fitted["output_weight"]
    )
    assert torch.equal(
        exact["indexed_phase_residual_output_bias"][8], fitted["output_bias"]
    )
