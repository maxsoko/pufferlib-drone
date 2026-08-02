from __future__ import annotations

import numpy as np
import torch

import scripts.train_vq2_lc212_stacked_adapter_output_ppo as target


def test_lc212_source_lock_and_actor_context() -> None:
    parent = target.verify_inputs()
    actor = target.split_actor_loader(parent, torch.device("cpu"))
    assert actor.initial_state(512, device="cpu").shape == (1, 512, 384)


def test_lc212_feature_decomposition() -> None:
    parent = target.verify_inputs()
    state = parent["model_state"]
    recurrent = torch.randn(1, 4, 384)
    index = torch.tensor([1, 3])
    feature = recurrent[0, index, -target.FEATURE_SIZE:]
    continuation = (
        feature @ state[target.OUTPUT_PARAMETER_NAMES[0]].T
        + state[target.OUTPUT_PARAMETER_NAMES[1]]
    )
    baseline = torch.randn(2, 4)
    current = baseline + continuation
    actual_feature, actual_baseline = target.output_feature_components(
        None, recurrent, current, index, state
    )
    assert torch.equal(actual_feature, feature)
    assert torch.allclose(actual_baseline, baseline, atol=2e-6, rtol=0.0)


def test_lc212_configuration_roundtrip() -> None:
    original_dtype = target.base.ROLLOUT_DTYPE
    snapshot = target.configure()
    try:
        assert target.base.TOTAL_AGENTS == target.TOTAL_AGENTS
        assert target.base.ROLLOUT_DTYPE == target.ROLLOUT_DTYPE
        assert target.base.ppo_update is target.ppo_output_update
        assert target.base.milestone.load_actor is target.split_actor_loader
    finally:
        target.restore(snapshot)
    assert target.base.ROLLOUT_DTYPE == original_dtype


def test_lc212_contract() -> None:
    assert target.TARGET_PHASE == 16
    assert target.TARGET_RAW_INDEX == 18
    assert target.ACTOR_BATCH_SIZE == 256
    assert target.ROLLOUT_DTYPE["hidden"].shape == (64,)
    assert np.all(np.asarray(target.EXPLORATION_STD) > 0.0)
