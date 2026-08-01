from __future__ import annotations

import torch

import scripts.eval_vq2_lc077_phase6_decoder_milestone as lc077


def test_candidate_states_have_exact_parent_and_endpoint() -> None:
    parent = lc077.verify_inputs()
    state = parent["model_state"]
    endpoint = lc077.endpoint_decoder()
    baseline = lc077.candidate_state_for_index(state, 0)
    final = lc077.candidate_state_for_index(state, len(lc077.ALPHAS) - 1)
    assert torch.equal(baseline["indexed_phase_residual_output"], state["indexed_phase_residual_output"])
    assert torch.equal(baseline["indexed_phase_residual_output_bias"], state["indexed_phase_residual_output_bias"])
    expected_weight = torch.lerp(
        state["indexed_phase_residual_output"][lc077.TARGET_PHASE],
        endpoint["output_weight"], lc077.ALPHAS[-1],
    )
    expected_bias = torch.lerp(
        state["indexed_phase_residual_output_bias"][lc077.TARGET_PHASE],
        endpoint["output_bias"], lc077.ALPHAS[-1],
    )
    assert torch.equal(final["indexed_phase_residual_output"][lc077.TARGET_PHASE], expected_weight)
    assert torch.equal(final["indexed_phase_residual_output_bias"][lc077.TARGET_PHASE], expected_bias)


def test_lc077_targets_index7_in_one_256_agent_vector() -> None:
    assert lc077.TARGET_PHASE == 6
    assert lc077.TARGET_RAW_INDEX == 7
    assert lc077.GROUP_SIZE * len(lc077.ALPHAS) == 256
    assert lc077.MAX_STEPS == 12_000

