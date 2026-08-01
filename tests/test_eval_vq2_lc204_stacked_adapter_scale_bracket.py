from __future__ import annotations

import torch

import scripts.eval_vq2_lc204_stacked_adapter_scale_bracket as target


def test_lc204_source_lock_and_zero_scale() -> None:
    parent = target.verify_inputs()
    zero = target.candidate_state_for_index(parent["model_state"], 0)
    assert set(zero) == set(parent["model_state"])
    for name, value in parent["model_state"].items():
        assert torch.equal(zero[name], value)


def test_lc204_only_scales_continuation_output() -> None:
    endpoint = target.fit_payload()["model_state"]
    candidate = target.candidate_state_for_index(
        target.prior.parent_payload()["model_state"], 2
    )
    scale = target.OUTPUT_SCALES[2]
    for name, value in endpoint.items():
        if name.startswith("continuation_adapter_output."):
            assert torch.equal(candidate[name], value * scale)
        else:
            assert torch.equal(candidate[name], value)


def test_lc204_exact_context_and_wrapper_roundtrip() -> None:
    assert target.GROUP_SIZE == 128
    assert target.PAIR_SIZE == 256
    assert target.OUTPUT_SCALES[0] == 0.0
    names = ("TAG", "GROUP_SIZE", "SCALE_PAIRS", "load_actor")
    before = tuple(getattr(target.prior, name) for name in names)
    snapshot = target.configure_wrapper()
    try:
        assert target.prior.TAG == target.TAG
        assert target.prior.load_actor is target.load_actor
    finally:
        target.restore(snapshot)
    assert tuple(getattr(target.prior, name) for name in names) == before
