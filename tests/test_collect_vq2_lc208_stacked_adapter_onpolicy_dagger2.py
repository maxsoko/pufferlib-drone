from __future__ import annotations

import torch

import scripts.collect_vq2_lc208_stacked_adapter_onpolicy_dagger2 as target


def test_lc208_source_lock_and_actor() -> None:
    parent = target.verify_inputs()
    actor = target.base.split_actor_loader(parent, torch.device("cpu"))
    assert all(child.initial_state(256, device="cpu").shape == (1, 256, 384) for child in actor.actors)


def test_lc208_outer_wrapper_roundtrip() -> None:
    names = ("TAG", "PARENT_CHECKPOINT", "verify_inputs", "corrected_writer")
    before = tuple(getattr(target.base, name) for name in names)
    snapshot = target.configure_outer()
    try:
        assert target.base.TAG == target.TAG
        assert target.base.corrected_writer is target.corrected_writer
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.base, name) for name in names) == before
