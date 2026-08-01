from __future__ import annotations

import torch

import scripts.eval_vq2_lc142_phase9_cem_milestone as lc142


def test_lc142_source_lock_and_candidate_state() -> None:
    parent = lc142.verify_inputs()
    candidate = lc142.candidate_payload()
    state = lc142.candidate_state_for_index(parent["model_state"], 1)
    assert all(torch.equal(state[name], candidate["model_state"][name]) for name in state)
    assert lc142.TARGET_PHASE == 9
    assert lc142.TARGET_RAW_INDEX == 10


def test_lc142_only_phase9_output_bias_changed() -> None:
    parent = lc142.verify_inputs()["model_state"]
    candidate = lc142.candidate_state_for_index(parent, 1)
    for name in parent:
        if name == "indexed_phase_residual_output_bias":
            keep = torch.arange(parent[name].shape[0]) != lc142.TARGET_PHASE
            assert torch.equal(parent[name][keep], candidate[name][keep])
        else:
            assert torch.equal(parent[name], candidate[name])


def test_lc142_pairwise_actor_batches_are_source_matched() -> None:
    lc142.configure()
    assert len(lc142.pairwise.BIASES) == 2
    assert lc142.milestone.TOTAL_AGENTS == 256
    assert lc142.pairwise.PAIR_SIZE == 256
