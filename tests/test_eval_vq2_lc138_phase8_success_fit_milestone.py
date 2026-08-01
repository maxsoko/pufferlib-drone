from __future__ import annotations

import torch

import scripts.eval_vq2_lc138_phase8_success_fit_milestone as lc138


def test_lc138_source_lock_and_candidate_states() -> None:
    parent = lc138.verify_inputs()
    parent_state = parent["model_state"]
    baseline = lc138.candidate_state_for_index(parent_state, 0)
    candidate = lc138.candidate_state_for_index(parent_state, 1)
    for name in lc138.base.PARAMETER_NAMES:
        assert torch.equal(baseline[name][8], parent_state[name][8])
        assert not torch.equal(candidate[name][8], parent_state[name][8])
        keep = torch.arange(parent_state[name].shape[0]) != 8
        assert torch.equal(candidate[name][keep], parent_state[name][keep])


def test_lc138_metadata_contract() -> None:
    baseline = lc138.candidate_metadata_for_index(0)
    candidate = lc138.candidate_metadata_for_index(1)
    assert baseline["endpoint_phases"] == []
    assert candidate["endpoint_phases"] == [8]
    assert candidate["parameter_delta_l2"] > 0.0
    assert lc138.TARGET_RAW_INDEX == 9
