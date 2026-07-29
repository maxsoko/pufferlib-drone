from __future__ import annotations

import scripts.collect_vq2_recurrent_dagger_next as collector
from scripts.collect_vq2_recurrent_dagger_gate2 import (
    AGENTS,
    CHECKPOINT_SHA256,
    COLLECTION_STEP_LIMIT,
    EPISODES,
    SCREEN_REPORT_SHA256,
    SEED,
    configure_collector_module,
)


def test_sf035_binds_sf033_gate2_failure_distribution() -> None:
    names = (
        "TAG",
        "DEFAULT_OUTPUT",
        "PREREGISTRATION",
        "CHECKPOINT",
        "CHECKPOINT_SHA256",
        "TRAIN_REPORT",
        "TRAIN_REPORT_SHA256",
        "SF023_REPORT",
        "SF023_REPORT_SHA256",
        "AGENTS",
        "EPISODES",
        "SEED",
        "COLLECTION_STEP_LIMIT",
        "load_sf022",
    )
    previous = {name: getattr(collector, name) for name in names}
    try:
        configure_collector_module()
        assert collector.CHECKPOINT_SHA256 == CHECKPOINT_SHA256
        assert collector.SF023_REPORT_SHA256 == SCREEN_REPORT_SHA256
        assert collector.AGENTS == collector.EPISODES == AGENTS == EPISODES == 512
        assert collector.SEED == SEED == 42035
        assert collector.COLLECTION_STEP_LIMIT == COLLECTION_STEP_LIMIT == 2048
    finally:
        for name, value in previous.items():
            setattr(collector, name, value)

