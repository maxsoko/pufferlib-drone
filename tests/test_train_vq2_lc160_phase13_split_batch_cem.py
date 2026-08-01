from __future__ import annotations

import scripts.train_vq2_lc160_phase13_split_batch_cem as lc160


def test_lc160_source_lock() -> None:
    parent = lc160.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc160.TARGET_PHASE == 13
    assert lc160.TARGET_RAW_INDEX == 14
    assert lc160.MAX_STEPS == 21_000


def test_lc160_configuration() -> None:
    originals = lc160.configure()
    try:
        assert lc160.prior.base.TARGET_PHASE == 13
        assert lc160.prior.base.TARGET_RAW_INDEX == 14
        assert lc160.prior.base.MAX_STEPS == 21_000
        assert lc160.GENERATIONS == 2
    finally:
        lc160.restore(originals)
