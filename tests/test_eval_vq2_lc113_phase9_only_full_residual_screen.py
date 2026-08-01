from __future__ import annotations

import torch

import scripts.eval_vq2_lc113_phase9_only_full_residual_screen as lc113


def test_lc113_source_lock_and_phase_isolation() -> None:
    parent = lc113.verify_inputs()["model_state"]
    candidate = lc113.candidate_state_for_index(parent, 3)
    assert lc113.ALPHAS == (0.0, 0.01, 0.03, 0.10)
    assert lc113.TARGET_PHASE == 9
    assert torch.equal(
        candidate["indexed_phase_residual_input"][8],
        parent["indexed_phase_residual_input"][8],
    )
    assert not torch.equal(
        candidate["indexed_phase_residual_input"][9],
        parent["indexed_phase_residual_input"][9],
    )
