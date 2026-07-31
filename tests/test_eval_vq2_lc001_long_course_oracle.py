from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_vq2_lc001_long_course_oracle import (
    EPISODE_SECONDS,
    LONG_COURSE_GATE_CAP,
    MIN_GATES,
    OFFICIAL_PROGRESS_SCALE,
    aggregate_reports,
    long_course_environment,
)


def test_long_course_environment_has_no_gate16_progress_ceiling() -> None:
    for count in (20, 24, LONG_COURSE_GATE_CAP):
        environment = long_course_environment(count)
        assert environment["num_gates"] == count
        assert environment["teacher_roll_until_gate_index"] == count
        assert environment["gate_radius"] == 0.75
        assert environment["time_limit_seconds"] == EPISODE_SECONDS
        assert environment["observable_gate_progress"] == 1
        assert environment["observable_gate_index_denominator"] == OFFICIAL_PROGRESS_SCALE
        assert environment["observable_gate_progress_unbounded"] == 1
        assert (count - 1) / OFFICIAL_PROGRESS_SCALE > 1.0


def test_long_course_environment_rejects_unsupported_counts() -> None:
    with pytest.raises(ValueError, match="gate count"):
        long_course_environment(MIN_GATES - 1)
    with pytest.raises(ValueError, match="gate count"):
        long_course_environment(LONG_COURSE_GATE_CAP + 1)


def _report(path: Path, *, gates: int, agents: int, wall: float) -> None:
    path.write_text(json.dumps({
        "num_gates": gates,
        "agents": agents,
        "episodes": agents,
        "native_steps": 100,
        "wall_time_seconds": wall,
        "admitted": True,
        "source_commit": "a" * 40,
        "source_sha256": {"source": "same"},
        "runtime": {
            "runtime": "same",
            "omp_num_threads": "1" if agents == 1 else "32",
            "omp_dynamic": "FALSE",
        },
        "safety": {"flight_sim_packets_sent": 0},
    }))


def test_aggregate_reports_measured_projected_speedup(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    vector20 = tmp_path / "vector20.json"
    vector24 = tmp_path / "vector24.json"
    _report(baseline, gates=20, agents=1, wall=2.0)
    _report(vector20, gates=20, agents=64, wall=8.0)
    _report(vector24, gates=24, agents=64, wall=9.0)
    report = aggregate_reports(
        baseline_path=baseline,
        vector20_path=vector20,
        vector24_path=vector24,
    )
    assert report["projected_wall_clock_speedup"] == 16.0
    assert report["throughput_gate_passed"] is True
    assert report["omp_contract_exact"] is True
    assert report["puffer_throughput_admitted"] is True
