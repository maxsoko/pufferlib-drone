from __future__ import annotations

import torch

import scripts.eval_vq2_lc088_phase7_parity_bias_scale_bracket as lc088


def test_lc088_parity_bracket_contract() -> None:
    assert lc088.SCALES == (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    assert lc088.GROUP_SIZE * len(lc088.SCALES) == 256
    assert lc088.TARGET_PHASE == 7
    assert lc088.TARGET_RAW_INDEX == 8
    assert lc088.MAX_STEPS == 12_000
    assert lc088.DIRECTION[1] > 0.0


def test_lc088_source_lock_and_mirrored_surgery() -> None:
    parent = lc088.verify_inputs()
    state = parent["model_state"]
    for index, scale in enumerate(lc088.SCALES):
        candidate = lc088.candidate_state_for_index(state, index)
        delta = (
            candidate["indexed_phase_residual_output_bias"][lc088.TARGET_PHASE]
            - state["indexed_phase_residual_output_bias"][lc088.TARGET_PHASE]
        )
        expected = torch.tensor([scale * value for value in lc088.DIRECTION])
        assert torch.allclose(delta, expected, atol=1e-8, rtol=0.0)


def test_lc088_does_not_change_other_heads() -> None:
    state = lc088.verify_inputs()["model_state"]
    candidate = lc088.candidate_state_for_index(state, 4)
    delta = (
        candidate["indexed_phase_residual_output_bias"]
        - state["indexed_phase_residual_output_bias"]
    )
    assert torch.count_nonzero(delta[:lc088.TARGET_PHASE]) == 0
    assert torch.count_nonzero(delta[lc088.TARGET_PHASE + 1:]) == 0
