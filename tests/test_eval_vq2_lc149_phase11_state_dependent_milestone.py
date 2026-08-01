from __future__ import annotations

import torch

import scripts.eval_vq2_lc149_phase11_state_dependent_milestone as lc149


def test_lc149_source_lock_and_candidate_state() -> None:
    parent = lc149.verify_inputs()
    candidate = lc149.candidate_payload()
    state = lc149.candidate_state_for_index(parent["model_state"], 1)
    assert all(torch.equal(state[name], candidate["model_state"][name]) for name in state)
    assert lc149.TARGET_PHASE == 11
    assert lc149.TARGET_RAW_INDEX == 12


def test_lc149_only_phase11_residual_parameters_changed() -> None:
    parent = lc149.verify_inputs()["model_state"]
    candidate = lc149.candidate_state_for_index(parent, 1)
    changed = []
    for name in parent:
        if name in lc149.PARAMETER_NAMES:
            keep = torch.arange(parent[name].shape[0]) != lc149.TARGET_PHASE
            assert torch.equal(parent[name][keep], candidate[name][keep])
            changed.append(not torch.equal(
                parent[name][lc149.TARGET_PHASE], candidate[name][lc149.TARGET_PHASE]
            ))
        else:
            assert torch.equal(parent[name], candidate[name])
    assert all(changed)


def test_lc149_pairwise_batches_and_lossless_selection() -> None:
    lc149.configure()
    assert lc149.milestone.TOTAL_AGENTS == 256
    assert lc149.pairwise.PAIR_SIZE == 256
    baseline = {
        "target_passes": 1, "transport_pass": True, "pre_target_terminals": 0,
        "paired_target_losses_vs_baseline": 0,
    }
    loss = {
        "target_passes": 2, "transport_pass": True, "pre_target_terminals": 0,
        "paired_target_losses_vs_baseline": 1,
    }
    gain = {**loss, "paired_target_losses_vs_baseline": 0}
    assert lc149.choose_candidate([baseline, loss]) is None
    assert lc149.choose_candidate([baseline, gain]) is gain
