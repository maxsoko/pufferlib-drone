from __future__ import annotations

import torch

import scripts.train_vq2_lc209_stacked_adapter_onpolicy_dagger2 as target


def test_lc209_source_lock_and_actor() -> None:
    parent = torch.load(
        target.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    actor = target.base.build_actor(parent)
    for name, value in parent["model_state"].items():
        assert torch.equal(actor.state_dict()[name], value)


def test_lc209_outer_wrapper_roundtrip() -> None:
    assert target.EPOCHS == 160
    names = ("TAG", "EPOCHS", "FEATURES", "verify_inputs", "materialize_sequences")
    before = tuple(getattr(target.base, name) for name in names)
    snapshot = target.configure_outer()
    try:
        assert target.base.TAG == target.TAG
        assert target.base.materialize_sequences is target.materialize_sequences
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.base, name) for name in names) == before
