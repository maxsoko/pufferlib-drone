from __future__ import annotations

import scripts.collect_vq2_lc155_phase11_failure_state_dagger2 as lc155


def test_lc155_source_lock() -> None:
    parent = lc155.verify_inputs()
    assert parent["numerically_admitted"]


def test_lc155_reuses_failure_state_dagger_contract() -> None:
    originals = lc155.configure()
    try:
        assert lc155.prior.PARENT_CHECKPOINT == lc155.PARENT_CHECKPOINT
        assert lc155.prior.TAG == lc155.TAG
        assert lc155.prior.GROUP_SIZE == 256
        assert lc155.prior.ENV_SEED_INDEX_OFFSET == 15
    finally:
        lc155.restore(originals)
