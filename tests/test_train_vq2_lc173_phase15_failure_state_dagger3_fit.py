from __future__ import annotations

import scripts.train_vq2_lc173_phase15_failure_state_dagger3_fit as lc173


def test_lc173_source_lock() -> None:
    parent = lc173.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc173.FEATURE_RECORDS > 0


def test_lc173_configuration() -> None:
    originals = lc173.configure()
    try:
        assert lc173.previous.PARENT_CHECKPOINT_SHA256 == lc173.PARENT_CHECKPOINT_SHA256
        assert lc173.previous.FIT_METADATA_FIELD == lc173.FIT_METADATA_FIELD
    finally:
        lc173.restore(originals)
