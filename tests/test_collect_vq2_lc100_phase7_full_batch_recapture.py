from __future__ import annotations

import scripts.collect_vq2_lc100_phase7_full_batch_recapture as lc100


def test_lc100_source_locks_lc099_phase7_outcomes() -> None:
    lc100.verify_inputs()


def test_lc100_exact_batch_contract() -> None:
    assert lc100.AGENTS == 256
    assert lc100.SEED_GROUP_SIZE == 128
    assert lc100.EXPECTED_QUERY_AGENTS == 8
    assert lc100.EXPECTED_SUCCESS_AGENTS == 2
    assert lc100.EXPECTED_FAILURE_AGENTS == 6
    assert lc100.MINIMUM_RECORDS == 10_000
