from __future__ import annotations

import scripts.train_vq2_lc137_phase8_success_fit_threshold as lc137


def test_lc137_source_lock() -> None:
    payload = lc137.verify_inputs()
    assert payload["numerically_admitted"]
    assert lc137.MINIMUM_SUCCESS_IMPROVEMENT == 1.4


def test_lc137_configuration_restores_fitter() -> None:
    before = (
        lc137.base.TAG,
        lc137.base.MINIMUM_SUCCESS_IMPROVEMENT,
        lc137.base.write_json_once,
    )
    originals = lc137.configure()
    try:
        assert lc137.base.TAG == lc137.TAG
        assert lc137.base.MINIMUM_SUCCESS_IMPROVEMENT == 1.4
        assert lc137.base.write_json_once is lc137.corrected_writer
    finally:
        lc137.restore(originals)
    after = (
        lc137.base.TAG,
        lc137.base.MINIMUM_SUCCESS_IMPROVEMENT,
        lc137.base.write_json_once,
    )
    assert after == before
