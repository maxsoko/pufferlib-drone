from __future__ import annotations

import torch

import scripts.train_vq2_lc141_phase9_split_batch_cem as lc141


def test_lc141_source_lock() -> None:
    parent = lc141.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc141.ACTOR_BATCH_SIZE == 256


def test_lc141_split_actor_shape_and_configuration_restore() -> None:
    parent = lc141.verify_inputs()
    before = (
        lc141.base.TAG, lc141.base.DEFAULT_OUTPUT,
        lc141.base.milestone.load_actor,
    )
    originals = lc141.configure()
    try:
        actor = lc141.base.milestone.load_actor(parent, torch.device("cpu"))
        assert len(actor.actors) == 2
        state = actor.initial_state(512, device="cpu")
        assert state.shape == (1, 512, 256)
        assert lc141.base.TAG == lc141.TAG
    finally:
        lc141.restore(originals)
    after = (
        lc141.base.TAG, lc141.base.DEFAULT_OUTPUT,
        lc141.base.milestone.load_actor,
    )
    assert after == before
