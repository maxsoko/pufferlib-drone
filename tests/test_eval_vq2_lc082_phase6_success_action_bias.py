from __future__ import annotations

import torch

import scripts.eval_vq2_lc082_phase6_success_action_bias as lc082


def test_lc082_uses_success_conditioned_signs_and_scale() -> None:
    assert lc082.BIASES[0][1] == (0.0, 0.0, 0.0, 0.0)
    assert lc082.BIASES[1][1][1] < 0.0
    assert lc082.BIASES[4][1][2] > 0.0
    assert lc082.BIASES[-1][1] == (0.010, -0.050, 0.025, 0.0)
    assert min(values[1] for _, values in lc082.BIASES) == -0.050


def test_lc082_exact_checkpoint_surgery() -> None:
    parent = lc082.verify_inputs()
    state = parent["model_state"]
    for index, (_, bias) in enumerate(lc082.BIASES):
        candidate = lc082.candidate_state_for_index(state, index)
        delta = (
            candidate["indexed_phase_residual_output_bias"][lc082.TARGET_PHASE]
            - state["indexed_phase_residual_output_bias"][lc082.TARGET_PHASE]
        )
        assert torch.allclose(delta, torch.tensor(bias), atol=1e-8, rtol=0.0)


def test_lc082_is_one_256_agent_raw_index7_vector() -> None:
    assert lc082.TARGET_PHASE == 6
    assert lc082.TARGET_RAW_INDEX == 7
    assert lc082.GROUP_SIZE * len(lc082.BIASES) == 256
    assert lc082.MAX_STEPS == 12_000
