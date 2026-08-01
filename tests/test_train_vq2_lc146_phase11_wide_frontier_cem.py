from __future__ import annotations

import numpy as np

import scripts.train_vq2_lc146_phase11_wide_frontier_cem as lc146


def test_lc146_source_lock() -> None:
    parent = lc146.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc146.GENERATIONS == 2


def test_lc146_wide_configuration_restores() -> None:
    before = (
        lc146.base.TAG, lc146.base.GENERATIONS,
        lc146.base.INITIAL_MEAN.copy(), lc146.base.INITIAL_STD.copy(),
        lc146.base.MAXIMUM_ABS_DELTA.copy(),
    )
    originals = lc146.configure()
    try:
        assert lc146.base.TAG == lc146.TAG
        assert lc146.base.GENERATIONS == 2
        assert np.array_equal(lc146.base.INITIAL_MEAN, lc146.INITIAL_MEAN)
        assert np.array_equal(lc146.base.INITIAL_STD, lc146.INITIAL_STD)
        assert np.array_equal(lc146.base.MAXIMUM_ABS_DELTA, lc146.MAXIMUM_ABS_DELTA)
    finally:
        lc146.restore(originals)
    after = (
        lc146.base.TAG, lc146.base.GENERATIONS,
        lc146.base.INITIAL_MEAN, lc146.base.INITIAL_STD,
        lc146.base.MAXIMUM_ABS_DELTA,
    )
    assert after[0] == before[0]
    assert after[1] == before[1]
    assert all(np.array_equal(left, right) for left, right in zip(after[2:], before[2:]))
