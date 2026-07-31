from __future__ import annotations

import torch

import scripts.fit_vq2_lc070_phase2_constrained_endpoint as lc070


def test_interpolate_endpoint_has_exact_endpoints() -> None:
    parent_weight = torch.randn(4, 64)
    parent_bias = torch.randn(4)
    target_weight = torch.randn(4, 64)
    target_bias = torch.randn(4)
    zero = lc070.interpolate_endpoint(
        parent_weight, parent_bias, target_weight, target_bias, 0.0
    )
    one = lc070.interpolate_endpoint(
        parent_weight, parent_bias, target_weight, target_bias, 1.0
    )
    assert torch.equal(zero[0], parent_weight)
    assert torch.equal(zero[1], parent_bias)
    assert torch.allclose(one[0], target_weight, atol=2e-7, rtol=0.0)
    assert torch.allclose(one[1], target_bias, atol=2e-7, rtol=0.0)


def test_grid_contains_baseline_and_small_feasible_scales() -> None:
    assert lc070.ALPHAS[0] == 0.0
    assert 0.1 in lc070.ALPHAS
    assert all(a < b for a, b in zip(lc070.ALPHAS, lc070.ALPHAS[1:]))

