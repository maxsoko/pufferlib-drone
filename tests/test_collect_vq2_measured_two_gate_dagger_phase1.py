from __future__ import annotations

import numpy as np

from scripts.collect_vq2_measured_two_gate_dagger_phase1 import (
    COLLECTION_STEP_LIMIT,
    configure_iteration,
    failure_state_collection_passes,
    iteration,
)


def test_sf058_retains_well_formed_failure_terminals() -> None:
    configure_iteration()
    assert iteration.COLLECTION_STEP_LIMIT == COLLECTION_STEP_LIMIT == 1600
    lengths = np.full(iteration.AGENTS, 900, dtype=np.int32)
    terminals = np.ones(iteration.AGENTS, dtype=np.int32)
    terminal_is_last = np.ones(iteration.AGENTS, dtype=bool)
    metrics = {
        "env/n": float(iteration.EPISODES),
        "env/timeout": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
        "env/out_of_order": 1.0,
    }
    assert failure_state_collection_passes(
        metrics,
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=terminal_is_last,
        labels=int(lengths.sum()),
        executed_action_max_error=0.0,
    )
