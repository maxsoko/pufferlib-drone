from __future__ import annotations

import torch

import scripts.eval_vq2_lc203_stacked_adapter_raw18_milestone as target


def test_lc203_source_lock_and_loader() -> None:
    parent = target.verify_inputs()
    candidate = target.candidate_payload()
    assert not candidate["numerically_admitted"]
    actor = target.load_actor(candidate, torch.device("cpu"))
    assert actor.continuation_phase_min == 16
    assert actor.continuation_phase_max_exclusive == 18
    assert actor.initial_state(2, device="cpu").shape == (1, 2, 384)
    for name, value in parent["model_state"].items():
        assert torch.equal(candidate["model_state"][name], value)


def test_lc203_exact_context_and_wrapper_roundtrip() -> None:
    assert target.prior.GROUP_SIZE == 128
    assert target.prior.PAIR_SIZE == 256
    names = ("TAG", "CANDIDATE_CHECKPOINT", "verify_inputs", "load_actor")
    before = tuple(getattr(target.prior, name) for name in names)
    snapshot = target.configure_wrapper()
    try:
        assert target.prior.TAG == target.TAG
        assert target.prior.load_actor is target.load_actor
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.prior, name) for name in names) == before
