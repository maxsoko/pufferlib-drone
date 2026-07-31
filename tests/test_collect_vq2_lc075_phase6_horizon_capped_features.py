from __future__ import annotations

import scripts.collect_vq2_lc075_phase6_horizon_capped_features as lc075


def test_lc075_changes_only_horizon_scale_and_identity() -> None:
    assert lc075.AGENTS == lc075.base.AGENTS == 256
    assert lc075.THREADS == lc075.base.THREADS == 32
    assert lc075.TARGET_PHASE == lc075.base.TARGET_PHASE == 6
    assert lc075.RECORD_STOP == lc075.base.RECORD_STOP == 20_000
    assert lc075.STEP_LIMIT == 12_000
    assert lc075.STEP_LIMIT == 2 * lc075.base.STEP_LIMIT


def test_lc074_rejection_is_source_locked() -> None:
    lc075.verify_inputs()

