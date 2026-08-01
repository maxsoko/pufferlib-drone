from __future__ import annotations

import torch

import scripts.eval_vq2_lc154_phase11_failure_state_dagger_milestone as lc154


def test_lc154_source_lock_and_candidate_state() -> None:
    parent = lc154.verify_inputs()
    candidate = lc154.candidate_payload()
    state = lc154.candidate_state_for_index(parent["model_state"], 1)
    assert all(torch.equal(state[name], candidate["model_state"][name]) for name in state)
    assert lc154.TARGET_RAW_INDEX == 12


def test_lc154_requires_complete_repeated_source_gain() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    partial = {
        "target_passes": 127, "paired_target_gains_vs_baseline": 127,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 1,
    }
    complete = {
        **partial, "target_passes": 128, "paired_target_gains_vs_baseline": 128,
        "pre_target_terminals": 0,
    }
    assert lc154.choose_candidate([baseline, partial]) is None
    assert lc154.choose_candidate([baseline, complete]) is complete
