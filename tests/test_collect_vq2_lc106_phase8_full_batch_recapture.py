from __future__ import annotations

import scripts.collect_vq2_lc106_phase8_full_batch_recapture as lc106


def test_lc106_source_lock_and_contract() -> None:
    lc106.verify_inputs()
    assert lc106.TARGET_PHASE == 8
    assert lc106.EXPECTED_QUERY_AGENTS == 4
    assert lc106.EXPECTED_SUCCESS_AGENTS == 2
    assert lc106.EXPECTED_FAILURE_AGENTS == 2
    assert lc106.MINIMUM_RECORDS == 4_000
    assert lc106.base.SEED_GROUP_SIZE == 128
