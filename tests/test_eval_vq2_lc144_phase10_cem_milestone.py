from __future__ import annotations

import torch

import scripts.eval_vq2_lc144_phase10_cem_milestone as lc144


def test_lc144_source_lock_and_candidate_state() -> None:
    parent = lc144.verify_inputs()
    candidate = lc144.candidate_payload()
    state = lc144.candidate_state_for_index(parent["model_state"], 1)
    assert all(torch.equal(state[name], candidate["model_state"][name]) for name in state)
    assert lc144.TARGET_PHASE == 10
    assert lc144.TARGET_RAW_INDEX == 11


def test_lc144_only_phase10_output_bias_changed() -> None:
    parent = lc144.verify_inputs()["model_state"]
    candidate = lc144.candidate_state_for_index(parent, 1)
    for name in parent:
        if name == "indexed_phase_residual_output_bias":
            keep = torch.arange(parent[name].shape[0]) != lc144.TARGET_PHASE
            assert torch.equal(parent[name][keep], candidate[name][keep])
        else:
            assert torch.equal(parent[name], candidate[name])


def test_lc144_pairwise_batches() -> None:
    lc144.configure()
    assert lc144.milestone.TOTAL_AGENTS == 256
    assert lc144.pairwise.PAIR_SIZE == 256
