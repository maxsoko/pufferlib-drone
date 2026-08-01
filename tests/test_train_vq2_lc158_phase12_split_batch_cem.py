from __future__ import annotations

import scripts.train_vq2_lc158_phase12_split_batch_cem as lc158


def test_lc158_source_lock() -> None:
    parent = lc158.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc158.TARGET_PHASE == 12
    assert lc158.TARGET_RAW_INDEX == 13


def test_lc158_configuration() -> None:
    originals = lc158.configure()
    try:
        assert lc158.prior.base.TARGET_PHASE == 12
        assert lc158.prior.base.TARGET_RAW_INDEX == 13
        assert lc158.GENERATIONS == 2
    finally:
        lc158.restore(originals)
