from __future__ import annotations

import torch

import scripts.eval_vq2_lc066_phase2_failure_conditioned as lc066


def test_decoder_surgery_equals_coefficient_times_failure_score() -> None:
    parent = lc066.verify_inputs()
    feature = torch.randn(64)
    weight, bias = lc066.failure_direction()
    for index in (0, 1, len(lc066.COEFFICIENTS) - 1):
        state = lc066.candidate_state_for_index(parent["model_state"], index)
        output_name = "indexed_phase_residual_output"
        bias_name = "indexed_phase_residual_output_bias"
        parent_action = (
            parent["model_state"][output_name][lc066.TARGET_PHASE] @ feature
            + parent["model_state"][bias_name][lc066.TARGET_PHASE]
        )
        candidate_action = (
            state[output_name][lc066.TARGET_PHASE] @ feature
            + state[bias_name][lc066.TARGET_PHASE]
        )
        coefficient = torch.tensor(lc066.COEFFICIENTS[index][1])
        expected = coefficient * (feature @ weight + bias)
        assert torch.allclose(candidate_action - parent_action, expected, atol=2e-6, rtol=0.0)


def test_lc066_is_one_fast_256_episode_vector() -> None:
    assert lc066.GROUP_SIZE * len(lc066.COEFFICIENTS) == 256
