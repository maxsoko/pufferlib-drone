from __future__ import annotations

from scripts.benchmark_vq2_lc003_native_puffer_training import (
    HORIZON,
    MEASURED_CYCLES,
    TOTAL_AGENTS,
    admission_predicates,
)


def test_lc003_admission_requires_real_training_speed_and_gpu_use() -> None:
    report = {
        "baseline_sha256": (
            "877b1d3086d9ed68fc17f10f31f58221492dde9a0183561e0e0b43d8f0a432e1"
        ),
        "measured_agent_steps": TOTAL_AGENTS * HORIZON * MEASURED_CYCLES,
        "puffer_log": {"loss": {"total": 1.0}},
        "end_to_end_speedup": 10.1,
        "gpu_sampling": {"maximum_percent": 80.0},
        "precision_bytes": 4,
        "deployment_legal": False,
        "checkpoints_written": 0,
        "flight_sim_packets_sent": 0,
        "submission_authorized": False,
    }
    assert all(admission_predicates(report).values())
    report["end_to_end_speedup"] = 9.99
    assert not admission_predicates(report)["speedup_at_least_10x"]
    report["end_to_end_speedup"] = 10.1
    report["gpu_sampling"]["maximum_percent"] = 49.0
    assert not admission_predicates(report)["gpu_reached_50_percent"]
