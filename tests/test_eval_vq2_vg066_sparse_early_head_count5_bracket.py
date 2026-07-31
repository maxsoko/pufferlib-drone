from __future__ import annotations

import torch

import scripts.eval_vq2_vg066_sparse_early_head_count5_bracket as vg066


def test_sparse_mapping_keeps_only_early_learned_heads() -> None:
    parent, update = vg066.load_endpoints()
    p = parent["model_state"]["indexed_phase_action_residual"]
    u = update["model_state"]["indexed_phase_action_residual"]
    assert torch.count_nonzero(p) == 0
    for phase in range(17):
        if phase in vg066.SPARSE_PHASES:
            assert torch.count_nonzero(u[phase]) > 0
        else:
            assert torch.count_nonzero(u[phase]) == 0


def test_count5_horizon_and_scale_ladder_are_fixed() -> None:
    assert vg066.NUM_GATES == 5
    assert vg066.MAX_STEPS == 360 * 64
    assert vg066.ALPHAS == (0.0, 0.25, 0.50, 0.75, 1.0)
    assert vg066.AGENTS == vg066.EPISODES == 64
