from __future__ import annotations

import torch

import scripts.eval_vq2_lc178_phase15_recurrent_adapter_milestone as lc178


def test_lc178_source_lock_and_frozen_base() -> None:
    parent = lc178.verify_inputs()
    candidate = lc178.candidate_payload()
    assert parent["numerically_admitted"]
    assert candidate["numerically_admitted"]
    for name, value in parent["model_state"].items():
        assert torch.equal(value, candidate["model_state"][name])


def test_lc178_loads_distinct_recurrent_contracts() -> None:
    parent_actor = lc178.load_actor(lc178.parent_payload(), torch.device("cpu"))
    candidate_actor = lc178.load_actor(lc178.candidate_payload(), torch.device("cpu"))
    parent_state = parent_actor.initial_state(3, device="cpu")
    candidate_state = candidate_actor.initial_state(3, device="cpu")
    assert parent_state.shape == (1, 3, 256)
    assert candidate_state.shape == (1, 3, 320)


def test_lc178_selection_requires_complete_gain() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    candidate = {
        "target_passes": 128,
        "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0,
        "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert lc178.choose_candidate([baseline, candidate]) is candidate
    candidate["target_passes"] = 127
    assert lc178.choose_candidate([baseline, candidate]) is None
