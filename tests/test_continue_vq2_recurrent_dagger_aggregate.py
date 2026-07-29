from __future__ import annotations

import scripts.train_vq2_recurrent_dagger_aggregate as aggregate
from scripts.continue_vq2_recurrent_dagger_aggregate import (
    CONFIG,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    SCHEMA,
    TAG,
    configure_aggregate_module,
)


def test_sf026_changes_only_parent_seed_epochs_and_learning_rate() -> None:
    assert CONFIG.seed == 42026
    assert CONFIG.epochs == 6
    assert CONFIG.learning_rate == 5e-5
    assert CONFIG.action_weights == (1.0, 1.0, 4.0, 1.0)
    assert aggregate.DAGGER_SOURCE_PATTERN == ("broad", "next", "next", "next")


def test_sf026_binds_frozen_aggregate_loop_to_new_lineage() -> None:
    configure_aggregate_module()
    assert aggregate.TAG == TAG
    assert aggregate.SCHEMA == SCHEMA
    assert aggregate.PARENT_CHECKPOINT == PARENT_CHECKPOINT
    assert aggregate.PARENT_CHECKPOINT_SHA256 == PARENT_CHECKPOINT_SHA256
