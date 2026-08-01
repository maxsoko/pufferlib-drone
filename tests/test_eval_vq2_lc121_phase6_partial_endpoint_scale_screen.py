from __future__ import annotations

import torch

import scripts.eval_vq2_lc121_phase6_partial_endpoint_scale_screen as lc121


def test_lc121_source_lock_and_scale_contract() -> None:
    parent = lc121.verify_inputs()
    assert lc121.ALPHAS == (0.0, 0.10, 0.30, 1.0)
    assert lc121.TARGET_PHASE == 6
    assert lc121.TARGET_RAW_INDEX == 10
    assert parent["schema"] == "vq2_lc105_phase7_endpoint_full_course_checkpoint_v1"


def test_lc121_changes_only_phase6_residual_rows() -> None:
    parent = lc121.verify_inputs()["model_state"]
    candidate = lc121.candidate_state_for_index(parent, 2)
    for name in lc121.PARAMETER_NAMES:
        assert not torch.equal(candidate[name][6], parent[name][6])
        assert torch.equal(candidate[name][:6], parent[name][:6])
        assert torch.equal(candidate[name][7:], parent[name][7:])
