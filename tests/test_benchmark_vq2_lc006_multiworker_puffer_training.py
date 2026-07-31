from __future__ import annotations

from scripts.benchmark_vq2_lc006_multiworker_puffer_training import WORKERS
from scripts.benchmark_vq2_lc003_native_puffer_training import (
    HORIZON,
    MEASURED_CYCLES,
    TOTAL_AGENTS,
)


def test_lc006_parallel_workload_exceeds_six_million_steps() -> None:
    assert WORKERS == 6
    assert WORKERS * TOTAL_AGENTS * HORIZON * MEASURED_CYCLES == 6_291_456
