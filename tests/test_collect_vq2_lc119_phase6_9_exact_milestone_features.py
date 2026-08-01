from __future__ import annotations

import numpy as np

import scripts.collect_vq2_lc119_phase6_9_exact_milestone_features as lc119


def test_lc119_source_lock_and_feature_contract() -> None:
    lc119.verify_inputs()
    assert lc119.EXPECTED_RECORDS == 60_082
    assert lc119.EXPECTED_QUERY_AGENTS == 19
    assert lc119.EXPECTED_SUCCESS_AGENTS == 4
    assert lc119.FEATURE_DTYPE.itemsize == 550


def test_lc119_outcomes_follow_last_record_terminal() -> None:
    records = np.zeros(4, dtype=lc119.FEATURE_DTYPE)
    records["agent_index"] = [128, 128, 129, 129]
    records["step"] = [10, 11, 10, 11]
    records["terminal"] = [0, 0, 0, 1]
    success, failure = lc119.outcome_agents(records)
    assert success.tolist() == [128]
    assert failure.tolist() == [129]
