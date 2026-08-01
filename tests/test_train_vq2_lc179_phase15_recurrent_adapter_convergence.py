from __future__ import annotations

import torch

import scripts.train_vq2_lc179_phase15_recurrent_adapter_convergence as lc179


def test_lc179_source_lock_and_actor() -> None:
    parent = lc179.verify_inputs()
    actor = lc179.build_actor(parent)
    assert parent["numerically_admitted"]
    for name, value in parent["model_state"].items():
        assert torch.equal(actor.state_dict()[name], value)


def test_lc179_convergence_contract() -> None:
    assert lc179.EPOCHS == 80
    assert lc179.LEARNING_RATE == 5e-4
    assert lc179.MINIMUM_VALIDATION_IMPROVEMENT == 10.0


def test_lc179_configuration_roundtrip() -> None:
    originals = lc179.configure()
    try:
        assert lc179.prior.EPOCHS == lc179.EPOCHS
        assert lc179.prior.build_actor is lc179.build_actor
    finally:
        lc179.restore(originals)
