from __future__ import annotations

import torch

import scripts.train_vq2_lc206_stacked_adapter_onpolicy_dagger as target


def test_lc206_source_lock_and_actor() -> None:
    parent = torch.load(target.PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    actor = target.build_actor(parent)
    for name, value in parent["model_state"].items():
        assert torch.equal(actor.state_dict()[name], value)
    assert actor.initial_state(2, device="cpu").shape == (1, 2, 384)


def test_lc206_configuration_roundtrip() -> None:
    assert target.EPOCHS == 120
    assert target.MAXIMUM_VALIDATION_MSE == 0.002
    original = target.prior.adapter_modules
    snapshot = target.configure()
    try:
        assert target.prior.adapter_modules is target.adapter_modules
        assert target.prior.TRAINABLE_ADAPTER_PREFIXES == target.TRAINABLE_PREFIXES
    finally:
        target.restore(snapshot)
    assert target.prior.adapter_modules is original
