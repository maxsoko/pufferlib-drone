from __future__ import annotations

import scripts.train_vq2_lc135_phase8_paired_rescue_fit as lc135


def test_lc135_source_lock() -> None:
    payload = lc135.verify_inputs()
    assert payload["numerically_admitted"]
    assert lc135.TARGET_PHASE == 8
    assert lc135.EXPECTED_QUERY_AGENTS == 6
    assert lc135.EXPECTED_SUCCESS_AGENTS == 3


def test_lc135_configuration_restores_fitter() -> None:
    before = (
        lc135.base.TAG, lc135.base.TARGET_PHASE,
        lc135.base.EXPECTED_QUERY_AGENTS, lc135.base.FEATURES,
    )
    originals = lc135.configure()
    try:
        assert lc135.base.TAG == lc135.TAG
        assert lc135.base.TARGET_PHASE == 8
        assert lc135.base.EXPECTED_QUERY_AGENTS == 6
        assert lc135.base.FEATURES == lc135.FEATURES
    finally:
        lc135.restore(originals)
    after = (
        lc135.base.TAG, lc135.base.TARGET_PHASE,
        lc135.base.EXPECTED_QUERY_AGENTS, lc135.base.FEATURES,
    )
    assert after == before
