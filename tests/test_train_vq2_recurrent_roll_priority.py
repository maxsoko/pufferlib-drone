from __future__ import annotations

import scripts.train_vq2_recurrent_dagger_aggregate as aggregate
from scripts.train_vq2_recurrent_roll_priority import (
    CONFIG,
    DIAGNOSIS_REPORT_SHA256,
    NEXT_REPORT_SHA256,
    PARENT_CHECKPOINT_SHA256,
    configure_aggregate_module,
)


def test_roll_priority_changes_loss_not_validation_contract() -> None:
    assert CONFIG.action_weights == (1.0, 1.0, 4.0, 1.0)
    assert CONFIG.loss_action_weights == (1.0, 4.0, 4.0, 1.0)
    assert CONFIG.prefer_admitted is True
    assert CONFIG.learning_rate == 2e-5
    assert CONFIG.epochs == 6


def test_roll_priority_source_locks_parent_data_and_diagnosis() -> None:
    assert PARENT_CHECKPOINT_SHA256.startswith("b7c0e1b0")
    assert NEXT_REPORT_SHA256.startswith("7c4d0d63")
    assert DIAGNOSIS_REPORT_SHA256.startswith("3c4a7f28")


def test_roll_priority_reuses_complete_next_prefix() -> None:
    old_pattern = aggregate.DAGGER_SOURCE_PATTERN
    try:
        configure_aggregate_module()
        assert aggregate.DAGGER_SOURCE_PATTERN == (
            "broad",
            "next",
            "next",
            "next",
        )
    finally:
        aggregate.DAGGER_SOURCE_PATTERN = old_pattern

