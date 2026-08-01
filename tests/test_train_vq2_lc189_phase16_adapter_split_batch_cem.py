from __future__ import annotations

import torch

import scripts.train_vq2_lc189_phase16_adapter_split_batch_cem as lc189


def test_lc189_source_lock_and_actor() -> None:
    parent = lc189.verify_inputs()
    actor = lc189.split_adapter_loader(parent, torch.device("cpu"))
    state = actor.initial_state(512, device="cpu")
    assert state.shape == (1, 512, 320)


def test_lc189_checkpoint_surgery_preserves_adapter() -> None:
    parent = lc189.verify_inputs()
    originals = lc189.configure()
    try:
        checkpoint, exact = lc189.base.checkpoint_with_delta(
            parent, torch.zeros(4).numpy()
        )
    finally:
        lc189.restore(originals)
    assert exact
    assert "phase_adapter_cell.weight_ih" in checkpoint["model_state"]


def test_lc189_search_contract() -> None:
    assert lc189.TARGET_PHASE == 16
    assert lc189.TARGET_RAW_INDEX == 17
    assert lc189.GENERATIONS == 2
