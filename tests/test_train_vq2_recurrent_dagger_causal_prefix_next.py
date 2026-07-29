from __future__ import annotations

import scripts.train_vq2_recurrent_dagger_aggregate as aggregate
from scripts.train_vq2_recurrent_dagger_causal_prefix_next import (
    CONFIG,
    NEXT_DATASET,
    NEXT_REPORT_SHA256,
    PARENT_CHECKPOINT,
    TAG,
    configure_aggregate_module,
)


def test_sf031_binds_new_prefix_and_preserves_record_ratio() -> None:
    names = (
        "TAG",
        "SCHEMA",
        "PREREGISTRATION",
        "PARENT_CHECKPOINT",
        "PARENT_CHECKPOINT_SHA256",
        "PARENT_REPORT",
        "PARENT_REPORT_SHA256",
        "NEXT_DATASET",
        "NEXT_REPORT_SHA256",
        "NEXT_METADATA_SHA256",
        "DAGGER_SOURCE_PATTERN",
    )
    previous = {name: getattr(aggregate, name) for name in names}
    try:
        configure_aggregate_module()
        assert aggregate.TAG == TAG
        assert aggregate.PARENT_CHECKPOINT == PARENT_CHECKPOINT
        assert aggregate.NEXT_DATASET == NEXT_DATASET
        assert aggregate.NEXT_REPORT_SHA256 == NEXT_REPORT_SHA256
        assert aggregate.DAGGER_SOURCE_PATTERN == (
            "broad",
            "next",
            "next",
            "next",
        )
    finally:
        for name, value in previous.items():
            setattr(aggregate, name, value)


def test_sf031_is_fixed_six_epoch_convergence_fit() -> None:
    assert CONFIG.seed == 42031
    assert CONFIG.epochs == 6
    assert CONFIG.learning_rate == 5e-5
    assert CONFIG.action_weights == (1.0, 1.0, 4.0, 1.0)
