from __future__ import annotations

import torch

import scripts.eval_vq2_lc207_stacked_adapter_onpolicy_milestone as target


def test_lc207_source_lock_and_loader() -> None:
    parent = target.verify_inputs()
    candidate = target.candidate_payload()
    actor = target.load_actor(candidate, torch.device("cpu"))
    assert actor.initial_state(2, device="cpu").shape == (1, 2, 384)
    for name, value in parent["model_state"].items():
        assert torch.equal(candidate["model_state"][name], value)


def test_lc207_outer_wrapper_roundtrip() -> None:
    names = ("TAG", "CANDIDATE_CHECKPOINT", "verify_inputs", "load_actor")
    before = tuple(getattr(target.base, name) for name in names)
    snapshot = target.configure_outer()
    try:
        assert target.base.TAG == target.TAG
        assert target.base.load_actor is target.load_actor
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.base, name) for name in names) == before
