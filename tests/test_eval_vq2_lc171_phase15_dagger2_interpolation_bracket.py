from __future__ import annotations

import torch

import scripts.eval_vq2_lc171_phase15_dagger2_interpolation_bracket as lc171


def test_lc171_source_lock_and_exact_fit() -> None:
    parent = lc171.verify_inputs()
    fitted = lc171.fit_payload()["model_state"]
    reconstructed = lc171.candidate_state_for_index(parent["model_state"], 4)
    assert all(torch.equal(reconstructed[name], fitted[name]) for name in fitted)


def test_lc171_requires_complete_gain() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    partial = {
        "target_passes": 127, "paired_target_gains_vs_baseline": 127,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 1, "fit_scale": 0.5,
    }
    assert lc171.choose_candidate([baseline, partial]) is None
