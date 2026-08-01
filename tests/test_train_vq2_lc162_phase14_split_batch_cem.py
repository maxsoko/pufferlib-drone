from __future__ import annotations

import scripts.train_vq2_lc162_phase14_split_batch_cem as lc162


def test_lc162_source_lock() -> None:
    parent = lc162.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc162.TARGET_PHASE == 14
    assert lc162.TARGET_RAW_INDEX == 15


def test_lc162_configuration() -> None:
    originals = lc162.configure()
    try:
        assert lc162.previous.prior.base.TARGET_PHASE == 14
        assert lc162.previous.prior.base.TARGET_RAW_INDEX == 15
        assert lc162.previous.prior.base.MAX_STEPS == 24_000
    finally:
        lc162.restore(originals)
