from __future__ import annotations

import numpy as np

import scripts.fit_vq2_lc101_phase7_success_anchored_endpoint as lc101


def test_lc101_source_lock_and_contract() -> None:
    lc101.verify_inputs()
    assert lc101.TARGET_PHASE == 7
    assert lc101.EXPECTED_QUERY_AGENTS == 8
    assert lc101.EXPECTED_SUCCESS_AGENTS == 2
    assert lc101.MINIMUM_FAILURE_IMPROVEMENT == 1.05
    assert lc101.MAXIMUM_SUCCESS_ACTION_DRIFT_MSE == 0.00025


def test_lc101_split_holds_one_success_in_each_partition() -> None:
    present = np.arange(8, dtype=np.int64)
    outcome = np.zeros(256, dtype=np.int8)
    outcome[present[-2:]] = 1
    lc101.configure()
    train, validation = lc101.base.stratified_agent_split(present, outcome)
    assert int((outcome[train] == 1).sum()) == 1
    assert int((outcome[validation] == 1).sum()) == 1
    assert not np.intersect1d(train, validation).size
    assert np.array_equal(np.sort(np.concatenate((train, validation))), present)
