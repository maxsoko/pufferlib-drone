from __future__ import annotations

import scripts.collect_vq2_lc172_phase15_failure_state_dagger3 as lc172


def test_lc172_source_lock() -> None:
    parent = lc172.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc172.prior.PHASE_MIN == 15


def test_lc172_configuration() -> None:
    originals = lc172.configure()
    try:
        assert lc172.prior.PARENT_CHECKPOINT_SHA256 == lc172.PARENT_CHECKPOINT_SHA256
        assert lc172.prior.FEATURE_SCHEMA == lc172.FEATURE_SCHEMA
    finally:
        lc172.restore(originals)


def test_lc172_base_writer_binding() -> None:
    assert callable(lc172.prior.prior.BASE_WRITE_JSON_ONCE)
