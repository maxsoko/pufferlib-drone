from __future__ import annotations

import scripts.collect_vq2_lc165_phase15_failure_state_dagger as lc165


def test_lc165_source_lock() -> None:
    parent = lc165.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc165.PHASE_MIN == 15
    assert lc165.TARGET_RAW_INDEX == 16


def test_lc165_configuration() -> None:
    originals = lc165.configure()
    try:
        assert lc165.prior.PHASE_MIN == 15
        assert lc165.prior.PHASE_MAX_EXCLUSIVE == 16
        assert lc165.prior.TARGET_RAW_INDEX == 16
    finally:
        lc165.restore(originals)
