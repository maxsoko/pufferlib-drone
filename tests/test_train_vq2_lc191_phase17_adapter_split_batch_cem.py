from __future__ import annotations

import scripts.train_vq2_lc191_phase17_adapter_split_batch_cem as lc191


def test_lc191_source_lock() -> None:
    assert lc191.verify_inputs()["frozen_non_phase16_state_exact"]


def test_lc191_contract_and_roundtrip() -> None:
    assert (lc191.TARGET_PHASE, lc191.TARGET_RAW_INDEX) == (17, 18)
    originals = lc191.configure()
    try:
        assert lc191.prior.PARENT_CHECKPOINT == lc191.PARENT_CHECKPOINT
    finally:
        lc191.restore(originals)
