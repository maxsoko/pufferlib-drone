from __future__ import annotations

import numpy as np

import scripts.train_vq2_lc166_phase15_failure_state_dagger_fit as lc166


def test_lc166_source_lock() -> None:
    parent = lc166.verify_inputs()
    assert parent["numerically_admitted"]
    assert len(lc166.SUCCESS_AGENTS) == 512
    assert lc166.TARGET_PHASE == 15


def test_lc166_configuration_uses_all_labeled_trajectories() -> None:
    originals = lc166.configure()
    try:
        assert lc166.prior.TARGET_PHASE == 15
        weights = lc166.prior.base.trajectory_weights(
            np.array((0, 0, 256, 256), dtype=np.int64),
            np.array((0, 256), dtype=np.int64),
        )
        assert np.isclose(weights.sum(), 1.0)
        assert lc166.prior.MAXIMUM_CONTROL_PARENT_DRIFT_MSE == 1.0
    finally:
        lc166.restore(originals)
