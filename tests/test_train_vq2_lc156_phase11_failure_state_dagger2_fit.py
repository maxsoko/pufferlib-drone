from __future__ import annotations

import scripts.train_vq2_lc156_phase11_failure_state_dagger2_fit as lc156


def test_lc156_source_lock() -> None:
    parent = lc156.verify_inputs()
    assert parent["numerically_admitted"]


def test_lc156_reuses_dagger_fit_contract() -> None:
    originals = lc156.configure()
    try:
        assert lc156.prior.PARENT_CHECKPOINT == lc156.PARENT_CHECKPOINT
        assert lc156.prior.FEATURES == lc156.FEATURES
        assert lc156.prior.FIT_METADATA_FIELD == lc156.FIT_METADATA_FIELD
        assert lc156.prior.MAXIMUM_CONTROL_PARENT_DRIFT_MSE == 1.0
    finally:
        lc156.restore(originals)
