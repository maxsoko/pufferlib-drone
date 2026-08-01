from __future__ import annotations

import torch

import scripts.train_vq2_lc180_phase15_recurrent_adapter_early_weighted as lc180


def test_lc180_source_lock_and_actor() -> None:
    parent = lc180.verify_inputs()
    actor = lc180.build_actor(parent)
    assert not parent["numerically_admitted"]
    for name, value in parent["model_state"].items():
        assert torch.equal(actor.state_dict()[name], value)


def test_lc180_weight_contract() -> None:
    assert lc180.EPOCHS == 100
    assert lc180.LEARNING_RATE == 1e-3
    assert lc180.TEMPORAL_WEIGHT_SEGMENTS == (
        (0, 64, 8.0), (64, 128, 12.0), (128, 256, 6.0), (256, 481, 2.0),
    )


def test_lc180_configuration_roundtrip() -> None:
    originals = lc180.configure()
    try:
        assert lc180.prior.TEMPORAL_WEIGHT_SEGMENTS == lc180.TEMPORAL_WEIGHT_SEGMENTS
        assert lc180.prior.build_actor is lc180.build_actor
    finally:
        lc180.restore(originals)
