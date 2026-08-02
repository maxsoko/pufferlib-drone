from __future__ import annotations

import torch

import scripts.train_vq2_lc211_lc209_phase16_cem as target


def test_lc211_source_lock_and_actor() -> None:
    parent = target.verify_inputs()
    actor = target.base.split_adapter_loader(parent, torch.device("cpu"))
    assert actor.initial_state(512, device="cpu").shape == (1, 512, 384)


def test_lc211_checkpoint_surgery_preserves_stacked_adapter() -> None:
    parent = target.verify_inputs()
    snapshot = target.configure_outer()
    try:
        inner = target.base.configure()
        try:
            checkpoint, exact = target.base.base.checkpoint_with_delta(
                parent, torch.zeros(4).numpy()
            )
        finally:
            target.base.restore(inner)
    finally:
        target.restore(snapshot)
    assert exact
    assert "continuation_adapter_cell.weight_ih" in checkpoint["model_state"]


def test_lc211_search_contract() -> None:
    assert target.TARGET_PHASE == 16
    assert target.TARGET_RAW_INDEX == 18
    assert target.GENERATIONS == 3
