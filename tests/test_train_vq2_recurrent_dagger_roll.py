from __future__ import annotations

from scripts.train_vq2_recurrent_dagger_roll import CONFIG, PARENT_SHA256, TAG


def test_roll_continuation_changes_only_seed_rate_epochs_and_roll_weight() -> None:
    assert TAG == "vq2_sf020_recurrent_dagger_roll_001"
    assert len(PARENT_SHA256) == 64
    assert CONFIG.seed == 42020
    assert CONFIG.epochs == 4
    assert CONFIG.learning_rate == 5e-5
    assert CONFIG.action_weights == (1.0, 2.0, 4.0, 1.0)
    assert CONFIG.agent_batch_size == 8
    assert CONFIG.sequence_chunk == 64
    assert CONFIG.smoothness_weight == 1e-4
