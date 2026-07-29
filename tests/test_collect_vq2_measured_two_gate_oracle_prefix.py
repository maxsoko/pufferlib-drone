from __future__ import annotations

from scripts.collect_vq2_measured_two_gate_oracle_prefix import (
    MEASURED_GATES,
    measured_config,
)


class _Puffer:
    pass


def test_measured_gate_geometry_is_source_locked() -> None:
    assert MEASURED_GATES == ((10.78, 0.084, 0.56), (25.13, 8.96, 1.65))

