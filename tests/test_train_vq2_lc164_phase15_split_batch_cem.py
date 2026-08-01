from __future__ import annotations

import scripts.train_vq2_lc164_phase15_split_batch_cem as lc164


def test_lc164_source_lock() -> None:
    parent = lc164.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc164.TARGET_PHASE == 15
    assert lc164.TARGET_RAW_INDEX == 16


def test_lc164_configuration() -> None:
    originals = lc164.configure()
    try:
        engine = lc164.previous.previous.prior.base
        assert engine.TARGET_PHASE == 15
        assert engine.TARGET_RAW_INDEX == 16
        assert engine.MAX_STEPS == 27_000
    finally:
        lc164.restore(originals)
