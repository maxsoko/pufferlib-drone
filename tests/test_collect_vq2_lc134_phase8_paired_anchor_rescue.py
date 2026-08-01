from __future__ import annotations

import scripts.collect_vq2_lc134_phase8_paired_anchor_rescue as lc134


def test_lc134_source_lock() -> None:
    payload = lc134.verify_inputs()
    assert payload["numerically_admitted"]
    assert lc134.TARGET_RAW_INDEX == 9
    assert lc134.PHASE_MIN == 8
    assert lc134.PHASE_MAX_EXCLUSIVE == 9


def test_lc134_configuration_restores_collector() -> None:
    before = (
        lc134.base.TAG, lc134.base.TARGET_RAW_INDEX,
        lc134.base.PHASE_MAX_EXCLUSIVE, lc134.base.CAPTURE_CONTROL_FEATURES,
    )
    originals = lc134.configure()
    try:
        assert lc134.base.TAG == lc134.TAG
        assert lc134.base.TARGET_RAW_INDEX == 9
        assert lc134.base.PHASE_MAX_EXCLUSIVE == 9
        assert lc134.base.CAPTURE_CONTROL_FEATURES
    finally:
        lc134.restore(originals)
    after = (
        lc134.base.TAG, lc134.base.TARGET_RAW_INDEX,
        lc134.base.PHASE_MAX_EXCLUSIVE, lc134.base.CAPTURE_CONTROL_FEATURES,
    )
    assert after == before
