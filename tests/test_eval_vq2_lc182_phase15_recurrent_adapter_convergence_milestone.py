from __future__ import annotations

import torch

import scripts.eval_vq2_lc182_phase15_recurrent_adapter_convergence_milestone as lc182


def test_lc182_source_lock_and_frozen_base() -> None:
    parent = lc182.verify_inputs()
    candidate = lc182.candidate_payload()
    assert parent["numerically_admitted"]
    assert candidate["numerically_admitted"]
    prefixes = ("phase_adapter_cell.", "phase_adapter_output.")
    for name, value in parent["model_state"].items():
        if not name.startswith(prefixes):
            assert torch.equal(value, candidate["model_state"][name])


def test_lc182_loads_equal_recurrent_contracts() -> None:
    parent_actor = lc182.load_actor(lc182.parent_payload(), torch.device("cpu"))
    candidate_actor = lc182.load_actor(lc182.candidate_payload(), torch.device("cpu"))
    assert parent_actor.initial_state(3, device="cpu").shape == (1, 3, 320)
    assert candidate_actor.initial_state(3, device="cpu").shape == (1, 3, 320)


def test_lc182_selection_requires_complete_gain() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    candidate = {
        "target_passes": 128,
        "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0,
        "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert lc182.choose_candidate([baseline, candidate]) is candidate
