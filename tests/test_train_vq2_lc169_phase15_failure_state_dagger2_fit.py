from __future__ import annotations

import scripts.train_vq2_lc169_phase15_failure_state_dagger2_fit as lc169


def test_lc169_source_lock() -> None:
    parent = lc169.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc169.previous.TARGET_PHASE == 15


def test_lc169_configuration() -> None:
    originals = lc169.configure()
    try:
        assert lc169.previous.PARENT_CHECKPOINT_SHA256 == lc169.PARENT_CHECKPOINT_SHA256
        assert lc169.previous.FIT_METADATA_FIELD == lc169.FIT_METADATA_FIELD
    finally:
        lc169.restore(originals)
