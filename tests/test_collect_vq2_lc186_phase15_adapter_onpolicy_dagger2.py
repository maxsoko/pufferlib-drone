from __future__ import annotations

import scripts.collect_vq2_lc186_phase15_adapter_onpolicy_dagger2 as lc186


def test_lc186_source_lock() -> None:
    parent = lc186.verify_inputs()
    assert parent["numerically_admitted"]
    assert parent["model"]["adapter_target_phase"] == 15


def test_lc186_configuration_roundtrip() -> None:
    original = lc186.prior.PARENT_CHECKPOINT
    originals = lc186.configure()
    try:
        assert lc186.prior.PARENT_CHECKPOINT == lc186.PARENT_CHECKPOINT
        assert lc186.prior.corrected_writer is lc186.corrected_writer
    finally:
        lc186.restore(originals)
    assert lc186.prior.PARENT_CHECKPOINT == original
