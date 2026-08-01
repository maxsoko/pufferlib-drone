from __future__ import annotations

import torch

import scripts.train_vq2_lc133_phase8_anisotropic_ppo as lc133


def test_lc133_source_lock_and_action_gap() -> None:
    payload = lc133.verify_inputs()
    assert payload["numerically_admitted"]
    gap = lc133.rescue_action_gap()
    assert gap["records"] == 1248
    assert abs(gap["action_rmse"] - 0.04401613728668076) < 1e-14
    assert gap["per_channel_rms"][3] < 0.001
    assert lc133.EXPLORATION_STD == (0.05, 0.05, 0.05, 0.002)


def test_lc133_configuration_restores_base_module() -> None:
    before = (
        lc133.base.TAG, lc133.base.TOTAL_AGENTS, lc133.base.EXPLORATION_STD,
        lc133.base.milestone.load_actor,
    )
    originals = lc133.configure()
    try:
        assert lc133.base.TAG == lc133.TAG
        assert lc133.base.TOTAL_AGENTS == 512
        assert lc133.base.EXPLORATION_STD == lc133.EXPLORATION_STD
        assert lc133.base.milestone.load_actor is lc133.split_actor_loader
        sample = torch.zeros(2, 4)
        scale = lc133.base.exploration_scale(sample, lc133.base.EXPLORATION_STD)
        assert torch.equal(scale, torch.tensor(lc133.EXPLORATION_STD))
    finally:
        lc133.restore(originals)
    after = (
        lc133.base.TAG, lc133.base.TOTAL_AGENTS, lc133.base.EXPLORATION_STD,
        lc133.base.milestone.load_actor,
    )
    assert after == before
