from __future__ import annotations

import torch

import scripts.train_vq2_lc202_phase16_17_stacked_adapter as lc202


def test_lc202_actor_freezes_lc189() -> None:
    parent = torch.load(
        lc202.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
    )
    actor = lc202.build_actor(parent)
    state = actor.state_dict()
    for name, value in parent["model_state"].items():
        assert torch.equal(state[name], value)
    assert torch.count_nonzero(state["continuation_adapter_output.weight"]) == 0
    assert torch.count_nonzero(state["continuation_adapter_output.bias"]) == 0


def test_lc202_contract_and_configuration_roundtrip() -> None:
    assert lc202.PHASE_MIN == 16
    assert lc202.PHASE_MAX_EXCLUSIVE == 18
    assert lc202.EPOCHS == 160
    original = lc202.prior.adapter_modules
    snapshot = lc202.configure()
    try:
        assert lc202.prior.adapter_modules is lc202.adapter_modules
        assert lc202.prior.TRAINABLE_ADAPTER_PREFIXES == lc202.TRAINABLE_PREFIXES
        parent = torch.load(
            lc202.PARENT_CHECKPOINT, map_location="cpu", weights_only=False
        )
        actor = lc202.build_actor(parent)
        contract = lc202.prior.fitted_model_contract(actor, parent["model"])
        assert contract["continuation_phase_min"] == 16
        assert contract["continuation_phase_max_exclusive"] == 18
    finally:
        lc202.restore(snapshot)
    assert lc202.prior.adapter_modules is original
