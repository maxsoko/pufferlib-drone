from __future__ import annotations

import torch

import scripts.eval_vq2_lc151_phase11_fit_endpoint_bracket as lc151


def test_lc151_source_lock_and_exact_fit_reconstruction() -> None:
    parent = lc151.verify_inputs()
    reconstructed = lc151.candidate_state_for_index(parent["model_state"], 1)
    fitted = lc151.fit_payload()["model_state"]
    assert all(torch.equal(reconstructed[name], fitted[name]) for name in fitted)
    assert lc151.FIT_SCALES == (0.0, 0.50, 0.75, 1.0, 1.25)


def test_lc151_changes_only_phase11_residual_rows() -> None:
    parent = lc151.verify_inputs()["model_state"]
    candidate = lc151.candidate_state_for_index(parent, 3)
    for name in parent:
        if name in lc151.PARAMETER_NAMES:
            keep = torch.arange(parent[name].shape[0]) != lc151.TARGET_PHASE
            assert torch.equal(parent[name][keep], candidate[name][keep])
            assert not torch.equal(
                parent[name][lc151.TARGET_PHASE], candidate[name][lc151.TARGET_PHASE]
            )
        else:
            assert torch.equal(parent[name], candidate[name])


def test_lc151_requires_complete_repeated_source_gain() -> None:
    baseline = {
        "target_passes": 0, "pre_target_terminals": 128,
    }
    partial = {
        "target_passes": 127, "paired_target_gains_vs_baseline": 127,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 1, "fit_scale": 0.75,
    }
    complete = {
        **partial, "target_passes": 128, "paired_target_gains_vs_baseline": 128,
        "pre_target_terminals": 0,
    }
    assert lc151.choose_candidate([baseline, partial]) is None
    assert lc151.choose_candidate([baseline, complete]) is complete
