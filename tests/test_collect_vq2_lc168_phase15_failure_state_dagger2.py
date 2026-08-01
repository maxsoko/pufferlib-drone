from __future__ import annotations

import scripts.collect_vq2_lc168_phase15_failure_state_dagger2 as lc168


def test_lc168_source_lock() -> None:
    parent = lc168.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc168.previous.PHASE_MIN == 15


def test_lc168_configuration() -> None:
    originals = lc168.configure()
    try:
        assert lc168.previous.PARENT_CHECKPOINT_SHA256 == lc168.PARENT_CHECKPOINT_SHA256
        assert lc168.previous.FEATURE_SCHEMA == lc168.FEATURE_SCHEMA
    finally:
        lc168.restore(originals)
