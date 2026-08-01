from __future__ import annotations

import numpy as np

import scripts.train_vq2_lc153_phase11_failure_state_dagger_fit as lc153


def test_lc153_source_lock() -> None:
    parent = lc153.verify_inputs()
    assert parent["numerically_admitted"]
    assert len(lc153.SUCCESS_AGENTS) == 512
    assert len(lc153.CONTROL_AGENTS) == 256
    assert lc153.TRAINING_SUCCESS_COUNT == 384


def test_lc153_configuration_uses_all_labeled_trajectories() -> None:
    originals = lc153.configure()
    try:
        assert lc153.base.TARGET_PHASE == 11
        assert lc153.base.SUCCESS_AGENTS == tuple(range(512))
        weights = lc153.base.trajectory_weights(
            np.array((0, 0, 256, 256), dtype=np.int64),
            np.array((0, 256), dtype=np.int64),
        )
        assert np.isclose(weights.sum(), 1.0)
        assert lc153.base.MAXIMUM_CONTROL_PARENT_DRIFT_MSE == 1.0
    finally:
        lc153.restore(originals)
