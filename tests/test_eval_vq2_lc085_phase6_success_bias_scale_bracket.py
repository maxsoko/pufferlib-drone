from __future__ import annotations

import torch

import scripts.eval_vq2_lc085_phase6_success_bias_scale_bracket as lc085


def test_lc085_scale_bracket_contract() -> None:
    assert lc085.SCALES == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    assert lc085.GROUP_SIZE * len(lc085.SCALES) == 256
    assert lc085.TARGET_PHASE == 6
    assert lc085.TARGET_RAW_INDEX == 7
    assert lc085.MAX_STEPS == 12_000


def test_lc085_source_lock_and_scaled_surgery() -> None:
    parent = lc085.verify_inputs()
    state = parent["model_state"]
    for index, scale in enumerate(lc085.SCALES):
        candidate = lc085.candidate_state_for_index(state, index)
        delta = (
            candidate["indexed_phase_residual_output_bias"][lc085.TARGET_PHASE]
            - state["indexed_phase_residual_output_bias"][lc085.TARGET_PHASE]
        )
        expected = torch.tensor([scale * value for value in lc085.DIRECTION])
        assert torch.allclose(delta, expected, atol=1e-8, rtol=0.0)
