from __future__ import annotations

import numpy as np

from scripts.collect_vq2_recurrent_dagger_causal_prefix_next import (
    AGENTS,
    COLLECTION_STEP_LIMIT,
    prefix_collection_passes,
)


def test_prefix_admission_accepts_clean_partial_survivors() -> None:
    lengths = np.full(AGENTS, COLLECTION_STEP_LIMIT, dtype=np.int32)
    lengths[:10] = 200
    terminals = np.zeros(AGENTS, dtype=np.int32)
    terminals[:10] = 1
    terminal_last = np.zeros(AGENTS, dtype=bool)
    terminal_last[:10] = True
    metrics = {
        "env/n": 10.0,
        "env/crash": 0.8,
        "env/timeout": 0.0,
        "env/out_of_order": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
    }
    assert prefix_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=terminal_last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
    lengths[-1] -= 1
    assert not prefix_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=terminal_last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
