from __future__ import annotations

import math

import scripts.train_vq2_lc175_phase15_alignment_return_ppo as lc175


def test_lc175_source_lock_and_gap() -> None:
    parent = lc175.verify_inputs()
    gap = lc175.source_measured_action_gap()
    assert parent["numerically_admitted"]
    assert gap["records"] == 123_136
    assert all(math.isfinite(value) and value > 0.0 for value in gap["per_channel_rms"])


def test_lc175_training_contract() -> None:
    assert lc175.TOTAL_AGENTS == 512
    assert lc175.ACTOR_BATCH_SIZE == 256
    assert lc175.SEED_GROUP_SIZE == 1
    assert lc175.SEED_INDEX_OFFSET == 15
    assert lc175.TARGET_PHASE == 15
    assert lc175.TARGET_RAW_INDEX == 16
    assert lc175.ROLLOUTS == 3
    assert lc175.PHASE_RETURN_ADVANTAGE_WEIGHT > 0.0


def test_lc175_configuration_roundtrip() -> None:
    originals = lc175.configure()
    try:
        assert lc175.base.MAX_STEPS == lc175.MAX_STEPS
        assert lc175.base.EXPLORATION_STD == lc175.EXPLORATION_STD
        assert lc175.base.milestone.load_actor is lc175.split_actor_loader
    finally:
        lc175.restore(originals)
