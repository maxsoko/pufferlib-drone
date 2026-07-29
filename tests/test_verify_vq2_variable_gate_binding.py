from __future__ import annotations

from scripts.verify_vq2_variable_gate_binding import (
    AGENTS,
    EPISODES_PER_AGENT,
    binding_distribution_passes,
)


def test_binding_distribution_requires_exact_uniform_counts() -> None:
    metrics = {"env/n": float(AGENTS * EPISODES_PER_AGENT)}
    for count in range(1, 17):
        metrics[f"env/gate_count{count}_episode"] = (
            0.125 if 5 <= count <= 12 else 0.0
        )
        metrics[f"env/gate_count{count}_success"] = 0.0
    assert binding_distribution_passes(metrics)
    metrics["env/gate_count6_episode"] -= 1.0 / 128.0
    assert not binding_distribution_passes(metrics)
