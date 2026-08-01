from __future__ import annotations

import torch

import scripts.eval_vq2_lc081_phase6_success_endpoint_milestone as lc081


def test_candidate_states_are_exact_parent_and_endpoint() -> None:
    parent = lc081.verify_inputs()
    state = parent["model_state"]
    endpoint = lc081.endpoint_decoder()
    baseline = lc081.candidate_state_for_index(state, 0)
    candidate = lc081.candidate_state_for_index(state, 1)
    assert torch.equal(
        baseline["indexed_phase_residual_output"],
        state["indexed_phase_residual_output"],
    )
    assert torch.equal(
        candidate["indexed_phase_residual_output"][lc081.TARGET_PHASE],
        endpoint["output_weight"],
    )
    assert torch.equal(
        candidate["indexed_phase_residual_output_bias"][lc081.TARGET_PHASE],
        endpoint["output_bias"],
    )


def test_lc081_is_one_paired_256_agent_vector() -> None:
    assert lc081.TARGET_PHASE == 6
    assert lc081.TARGET_RAW_INDEX == 7
    assert lc081.GROUP_SIZE == 128
    assert lc081.GROUP_SIZE * len(lc081.ALPHAS) == 256
    assert lc081.MAX_STEPS == 12_000
