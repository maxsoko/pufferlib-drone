from __future__ import annotations

import torch

import scripts.train_vq2_lc181_phase15_recurrent_adapter_final_convergence as lc181


def test_lc181_source_lock_and_actor() -> None:
    parent = lc181.verify_inputs()
    actor = lc181.build_actor(parent)
    assert not parent["numerically_admitted"]
    for name, value in parent["model_state"].items():
        assert torch.equal(actor.state_dict()[name], value)


def test_lc181_terminal_convergence_contract() -> None:
    assert lc181.EPOCHS == 300
    assert lc181.MINIMUM_VALIDATION_IMPROVEMENT == 4.0
    assert lc181.MAXIMUM_VALIDATION_MSE == 4e-5


def test_lc181_configuration_roundtrip() -> None:
    originals = lc181.configure()
    try:
        assert lc181.prior.MAXIMUM_VALIDATION_MSE == lc181.MAXIMUM_VALIDATION_MSE
        assert lc181.prior.build_actor is lc181.build_actor
    finally:
        lc181.restore(originals)
