from __future__ import annotations

import torch

import scripts.collect_vq2_lc147_phase11_state_dependent_rescue as lc147


def test_lc147_source_lock() -> None:
    parent = lc147.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc147.TARGET_RAW_INDEX == 12
    assert lc147.PHASE_MIN == 11


def test_lc147_configuration_restores() -> None:
    before = (
        lc147.base.TAG, lc147.base.GROUP_SIZE,
        lc147.base.ENV_SEED_GROUP_SIZE, lc147.base.ENV_SEED_INDEX_OFFSET,
        lc147.base.milestone.load_actor,
    )
    parent = lc147.verify_inputs()
    originals = lc147.configure()
    try:
        assert lc147.base.TOTAL_AGENTS == 512
        assert lc147.base.ENV_SEED_GROUP_SIZE == 1
        assert lc147.base.ENV_SEED_INDEX_OFFSET == 15
        actor = lc147.base.milestone.load_actor(parent, torch.device("cpu"))
        assert len(actor.actors) == 2
    finally:
        lc147.restore(originals)
    after = (
        lc147.base.TAG, lc147.base.GROUP_SIZE,
        lc147.base.ENV_SEED_GROUP_SIZE, lc147.base.ENV_SEED_INDEX_OFFSET,
        lc147.base.milestone.load_actor,
    )
    assert after == before
