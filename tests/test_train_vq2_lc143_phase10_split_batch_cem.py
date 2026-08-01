from __future__ import annotations

import numpy as np
import torch

import scripts.train_vq2_lc143_phase10_split_batch_cem as lc143


def test_lc143_source_lock() -> None:
    parent = lc143.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc143.TARGET_PHASE == 10
    assert lc143.TARGET_RAW_INDEX == 11
    assert lc143.MAX_STEPS == 15_000


def test_lc143_configuration_and_phase10_surgery() -> None:
    parent = lc143.verify_inputs()
    originals = lc143.configure()
    try:
        assert lc143.base.TARGET_PHASE == 10
        assert lc143.base.TARGET_RAW_INDEX == 11
        actor = lc143.base.milestone.load_actor(parent, torch.device("cpu"))
        assert len(actor.actors) == 2
        delta = np.array((0.01, -0.02, 0.03, -0.004), dtype=np.float32)
        checkpoint, frozen = lc143.base.checkpoint_with_delta(parent, delta)
        assert frozen
        assert checkpoint[lc143.FROZEN_STATE_FIELD]
        assert checkpoint[lc143.DELTA_FIELD] == delta.tolist()
        before = parent["model_state"]
        after = checkpoint["model_state"]
        for name in before:
            if name == lc143.base.PHASE_OUTPUT_BIAS:
                keep = torch.arange(before[name].shape[0]) != lc143.TARGET_PHASE
                assert torch.equal(before[name][keep], after[name][keep])
            else:
                assert torch.equal(before[name], after[name])
    finally:
        lc143.restore(originals)
