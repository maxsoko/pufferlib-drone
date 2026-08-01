from __future__ import annotations

import numpy as np

import scripts.train_vq2_lc148_phase11_state_dependent_fit as lc148


def test_lc148_source_lock() -> None:
    parent = lc148.verify_inputs()
    assert parent["numerically_admitted"]
    assert len(lc148.SUCCESS_AGENTS) == 256
    assert len(lc148.CONTROL_AGENTS) == 256
    assert lc148.TRAINING_SUCCESS_COUNT == 192


def test_lc148_configuration_restores() -> None:
    before = (
        lc148.base.TAG, lc148.base.TARGET_PHASE,
        lc148.base.SUCCESS_AGENTS, lc148.base.CONTROL_AGENTS,
    )
    originals = lc148.configure()
    try:
        assert lc148.base.TARGET_PHASE == 11
        assert lc148.base.SUCCESS_AGENTS == tuple(range(256, 512))
        weights = lc148.base.trajectory_weights(
            np.array((256, 256, 257, 257), dtype=np.int64),
            np.array((256, 257), dtype=np.int64),
        )
        assert np.isclose(weights.sum(), 1.0)
    finally:
        lc148.restore(originals)
    after = (
        lc148.base.TAG, lc148.base.TARGET_PHASE,
        lc148.base.SUCCESS_AGENTS, lc148.base.CONTROL_AGENTS,
    )
    assert after == before
