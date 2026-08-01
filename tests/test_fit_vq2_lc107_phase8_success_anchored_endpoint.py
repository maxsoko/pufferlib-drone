from __future__ import annotations

import numpy as np

import scripts.fit_vq2_lc107_phase8_success_anchored_endpoint as lc107


def test_lc107_source_lock_and_contract() -> None:
    lc107.verify_inputs()
    assert lc107.TARGET_PHASE == 8
    assert lc107.EXPECTED_QUERY_AGENTS == 4
    assert lc107.EXPECTED_SUCCESS_AGENTS == 2
    assert lc107.MINIMUM_FAILURE_IMPROVEMENT == 1.05


def test_lc107_split_holds_one_agent_from_each_class() -> None:
    present = np.arange(4, dtype=np.int64)
    outcome = np.zeros(256, dtype=np.int8)
    outcome[present[-2:]] = 1
    lc107.configure()
    train, validation = lc107.base.stratified_agent_split(present, outcome)
    assert train.size == validation.size == 2
    assert int((outcome[train] == 1).sum()) == 1
    assert int((outcome[validation] == 1).sum()) == 1
