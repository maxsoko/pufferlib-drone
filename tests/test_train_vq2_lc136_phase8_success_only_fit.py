from __future__ import annotations

import numpy as np

import scripts.train_vq2_lc136_phase8_success_only_fit as lc136


def test_lc136_source_lock_and_trajectory_weights() -> None:
    payload = lc136.verify_inputs()
    assert payload["numerically_admitted"]
    agents = np.asarray([3, 3, 3, 7, 7], dtype=np.int64)
    weights = lc136.trajectory_weights(agents, np.asarray([3, 7]))
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[agents == 3].sum(), 0.5)
    assert np.isclose(weights[agents == 7].sum(), 0.5)


def test_lc136_success_only_contract() -> None:
    assert lc136.TARGET_PHASE == 8
    assert lc136.MINIMUM_SUCCESS_IMPROVEMENT == 2.0
    assert lc136.MAXIMUM_CONTROL_PARENT_DRIFT_MSE == 0.02
    assert lc136.SCALES == (0.10, 0.30, 0.50, 1.0)
