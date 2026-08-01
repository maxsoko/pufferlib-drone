from __future__ import annotations

import numpy as np

from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import FEATURE_DTYPE
import scripts.train_vq2_lc187_phase15_adapter_onpolicy_dagger2 as lc187


def test_lc187_source_and_sequence_lock() -> None:
    assert lc187.verify_inputs()["numerically_admitted"]
    records = np.memmap(lc187.FEATURES, dtype=FEATURE_DTYPE, mode="r")
    assert lc187.sequence_contract(records) == lc187.EXPECTED_SEQUENCE_CONTRACT


def test_lc187_fit_contract() -> None:
    assert lc187.EPOCHS == 160
    assert lc187.LEARNING_RATE == 5e-4


def test_lc187_configuration_roundtrip() -> None:
    original = lc187.prior.materialize_sequences
    originals = lc187.configure()
    try:
        assert lc187.prior.materialize_sequences is lc187.materialize_sequences
    finally:
        lc187.restore(originals)
    assert lc187.prior.materialize_sequences is original
