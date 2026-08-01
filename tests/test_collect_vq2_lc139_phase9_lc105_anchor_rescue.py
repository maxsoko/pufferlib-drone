from __future__ import annotations

import scripts.collect_vq2_lc139_phase9_lc105_anchor_rescue as lc139


def test_lc139_source_lock() -> None:
    payload = lc139.verify_inputs()
    assert payload["numerically_admitted"]
    assert lc139.TARGET_RAW_INDEX == 10
    assert lc139.PHASE_MIN == 9
    assert lc139.PHASE_MAX_EXCLUSIVE == 10


def test_lc139_configuration_restores_collector() -> None:
    before = (
        lc139.base.TAG, lc139.base.PARENT_CHECKPOINT,
        lc139.base.PHASE_MIN, lc139.base.CAPTURE_CONTROL_FEATURES,
    )
    originals = lc139.configure()
    try:
        assert lc139.base.TAG == lc139.TAG
        assert lc139.base.PARENT_CHECKPOINT == lc139.PARENT_CHECKPOINT
        assert lc139.base.PHASE_MIN == 9
        assert lc139.base.CAPTURE_CONTROL_FEATURES
    finally:
        lc139.restore(originals)
    after = (
        lc139.base.TAG, lc139.base.PARENT_CHECKPOINT,
        lc139.base.PHASE_MIN, lc139.base.CAPTURE_CONTROL_FEATURES,
    )
    assert after == before
