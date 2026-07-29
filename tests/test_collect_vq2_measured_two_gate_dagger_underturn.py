from __future__ import annotations

import numpy as np

from scripts.collect_vq2_measured_two_gate_dagger_underturn import (
    COLLECTION_STEP_LIMIT,
    configure_iteration,
    failure_state_collection_passes,
    iteration,
)


def test_sf065_retains_well_formed_safe_miss_terminals() -> None:
    configure_iteration()
    assert iteration.COLLECTION_STEP_LIMIT == COLLECTION_STEP_LIMIT == 1600
    assert iteration.SEED == 42065
    lengths = np.full(iteration.AGENTS, 462, dtype=np.int32)
    terminals = np.ones(iteration.AGENTS, dtype=np.int32)
    terminal_is_last = np.ones(iteration.AGENTS, dtype=bool)
    metrics = {
        "env/n": float(iteration.EPISODES),
        "env/timeout": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
        "env/missed_gate": 1.0,
    }
    assert failure_state_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=terminal_is_last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
