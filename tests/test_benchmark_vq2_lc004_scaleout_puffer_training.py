from __future__ import annotations

from scripts.benchmark_vq2_lc004_scaleout_puffer_training import (
    HORIZON,
    MINIBATCH_SIZE,
    NUM_BUFFERS,
    NUM_THREADS,
    TOTAL_AGENTS,
)


def test_lc004_uses_the_full_host_without_changing_horizon() -> None:
    assert TOTAL_AGENTS == 4096
    assert NUM_BUFFERS == 64
    assert NUM_THREADS == 256
    assert HORIZON == 16
    assert MINIBATCH_SIZE == TOTAL_AGENTS * HORIZON
    assert TOTAL_AGENTS % NUM_BUFFERS == 0
    assert NUM_THREADS % NUM_BUFFERS == 0
