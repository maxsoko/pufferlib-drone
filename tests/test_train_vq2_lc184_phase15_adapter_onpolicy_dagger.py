from __future__ import annotations

import numpy as np
import torch

from scripts.collect_vq2_lc119_phase6_9_exact_milestone_features import FEATURE_DTYPE
import scripts.train_vq2_lc184_phase15_adapter_onpolicy_dagger as lc184


def test_lc184_source_lock_and_actor() -> None:
    parent = lc184.verify_inputs()
    actor = lc184.build_actor(parent)
    assert parent["numerically_admitted"]
    for name, value in parent["model_state"].items():
        assert torch.equal(actor.state_dict()[name], value)


def test_lc184_sequence_contract() -> None:
    records = np.memmap(lc184.FEATURES, dtype=FEATURE_DTYPE, mode="r")
    assert lc184.sequence_contract(records) == lc184.EXPECTED_SEQUENCE_CONTRACT


def test_lc184_fit_contract() -> None:
    assert lc184.EPOCHS == 120
    assert lc184.MINIMUM_VALIDATION_IMPROVEMENT == 2.0
    assert lc184.MAXIMUM_VALIDATION_MSE == 2e-4


def test_lc184_configuration_roundtrip() -> None:
    original = lc184.prior.materialize_sequences
    originals = lc184.configure()
    try:
        assert lc184.prior.materialize_sequences is lc184.materialize_sequences
        assert lc184.prior.PARENT_CHECKPOINT_SHA256 == lc184.PARENT_CHECKPOINT_SHA256
    finally:
        lc184.restore(originals)
    assert lc184.prior.materialize_sequences is original
