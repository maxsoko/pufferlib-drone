from __future__ import annotations

import torch

import scripts.eval_vq2_lc071_phase2_constrained_endpoint_screen as lc071


def test_candidate_states_have_exact_baseline_and_endpoint() -> None:
    parent = lc071.verify_inputs()
    state = parent["model_state"]
    endpoint = lc071.endpoint_decoder()
    baseline = lc071.candidate_state_for_index(state, 0)
    selected = lc071.candidate_state_for_index(state, len(lc071.ALPHAS) - 1)
    output_name = "indexed_phase_residual_output"
    bias_name = "indexed_phase_residual_output_bias"
    assert torch.equal(baseline[output_name], state[output_name])
    assert torch.equal(baseline[bias_name], state[bias_name])
    assert torch.equal(
        selected[output_name][lc071.TARGET_PHASE], endpoint[0]
    )
    assert torch.equal(
        selected[bias_name][lc071.TARGET_PHASE], endpoint[1]
    )


def test_lc071_is_one_fast_256_episode_vector() -> None:
    assert lc071.GROUP_SIZE * len(lc071.ALPHAS) == 256
    assert lc071.ALPHAS[0] == 0.0
    assert lc071.ALPHAS[-1] == 1.0

