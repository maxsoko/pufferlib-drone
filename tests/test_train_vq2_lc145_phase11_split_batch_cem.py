from __future__ import annotations

import numpy as np
import torch

import scripts.train_vq2_lc145_phase11_split_batch_cem as lc145


def test_lc145_source_lock() -> None:
    parent = lc145.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc145.TARGET_PHASE == 11
    assert lc145.TARGET_RAW_INDEX == 12
    assert lc145.MAX_STEPS == 17_000


def test_lc145_configuration_and_phase11_surgery() -> None:
    parent = lc145.verify_inputs()
    originals = lc145.configure()
    try:
        actor = lc145.base.milestone.load_actor(parent, torch.device("cpu"))
        assert len(actor.actors) == 2
        delta = np.array((0.01, -0.02, 0.03, -0.004), dtype=np.float32)
        checkpoint, frozen = lc145.base.checkpoint_with_delta(parent, delta)
        assert frozen and checkpoint[lc145.FROZEN_STATE_FIELD]
        assert checkpoint[lc145.DELTA_FIELD] == delta.tolist()
        before, after = parent["model_state"], checkpoint["model_state"]
        for name in before:
            if name == lc145.base.PHASE_OUTPUT_BIAS:
                keep = torch.arange(before[name].shape[0]) != lc145.TARGET_PHASE
                assert torch.equal(before[name][keep], after[name][keep])
            else:
                assert torch.equal(before[name], after[name])
    finally:
        lc145.restore(originals)
