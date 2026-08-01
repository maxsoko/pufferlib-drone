from __future__ import annotations

import torch

import scripts.eval_vq2_lc122_phase6_8_composite_scale_screen as lc122


def test_lc122_source_lock_and_composite_contract() -> None:
    parent = lc122.verify_inputs()["model_state"]
    assert lc122.PHASE6_SCALE == 0.30
    assert lc122.PHASE8_SCALES == (0.0, 0.10, 0.30, 1.0)
    baseline = lc122.candidate_state_for_index(parent, 0)
    candidate = lc122.candidate_state_for_index(parent, 2)
    assert all(torch.equal(baseline[name], parent[name]) for name in lc122.PARAMETER_NAMES)
    for name in lc122.PARAMETER_NAMES:
        assert not torch.equal(candidate[name][6], parent[name][6])
        assert not torch.equal(candidate[name][8], parent[name][8])
        assert torch.equal(candidate[name][7], parent[name][7])
        assert torch.equal(candidate[name][9], parent[name][9])
