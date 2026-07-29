from __future__ import annotations

import scripts.collect_vq2_public_phase_dagger_next as collector
from scripts.collect_vq2_public_phase_prefix_next import (
    CHECKPOINT,
    SEED,
    configure_collector,
)


def test_sf046_configures_fresh_phase_actor_collection() -> None:
    configure_collector()
    assert collector.SEED == SEED == 42046
    assert collector.CHECKPOINT == CHECKPOINT
    assert collector.COLLECTION_STEP_LIMIT == 768

