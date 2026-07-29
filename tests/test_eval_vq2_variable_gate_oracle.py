from __future__ import annotations

import pytest

from scripts.eval_vq2_variable_gate_oracle import (
    ACCEPTANCE_COUNTS,
    EPISODE_SECONDS,
    MINIMUM_SUCCESS_RATE,
    mixed_variable_environment,
    variable_environment,
    variable_oracle_passes,
)


def _passing_metrics(num_gates: int, success_rate: float = 1.0) -> dict[str, float]:
    metrics = {
        "env/n": 512.0,
        "env/success_rate": success_rate,
        "env/crash": 0.0,
        "env/out_of_order": 0.0,
        "env/crossing_margin_violation": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
        f"env/gate_count{num_gates}_episode": 1.0,
        f"env/gate_count{num_gates}_success": success_rate,
    }
    for gate in range(num_gates):
        metrics[f"env/ordered_gate{gate}_sampled"] = success_rate
        metrics[f"env/ordered_gate{gate}_radial"] = 0.05
    return metrics


def test_variable_oracle_contract_is_full_course_and_count_agnostic() -> None:
    assert ACCEPTANCE_COUNTS == (5, 8, 11, 12)
    assert MINIMUM_SUCCESS_RATE == 0.99
    assert EPISODE_SECONDS == 360.0
    for count in range(5, 13):
        environment = variable_environment(count)
        assert environment["num_gates"] == count
        assert environment["gate_radius"] == 0.75
        assert environment["teacher_alignment_governor"] == 1
        assert environment["teacher_action_blend"] == 1.0
        assert environment["teacher_roll_until_gate_index"] == count
        assert environment["observable_gate_index_denominator"] == 16.0
        assert environment["gate_position_require_valid_course"] == 1
        assert environment["gate_position_min_forward_gap_m"] == 8.0
        assert environment["gate_position_max_segment_distance_m"] == 40.0
        assert environment["sitl_plant_domain_randomize"] == 0


def test_variable_environment_rejects_out_of_range_count() -> None:
    with pytest.raises(ValueError, match=r"\[5, 12\]"):
        variable_environment(4)
    with pytest.raises(ValueError, match=r"\[5, 12\]"):
        variable_environment(13)


def test_mixed_profile_samples_per_instance_and_retains_six_anchor() -> None:
    environment = mixed_variable_environment(seed=429021)
    assert environment["num_gates"] == 6
    assert environment["num_gates_per_env_randomize"] == 1
    assert environment["num_gates_per_env_min"] == 5
    assert environment["num_gates_per_env_max"] == 12
    assert environment["num_gates_per_env_seed"] == 429021
    assert environment["teacher_roll_until_gate_index"] == 16


def test_variable_oracle_admits_99_percent_but_never_collision() -> None:
    success_rate = 507.0 / 512.0
    metrics = _passing_metrics(12, success_rate)
    assert variable_oracle_passes(metrics, episodes=512, num_gates=12)
    metrics["env/crash"] = 1.0 / 512.0
    assert not variable_oracle_passes(metrics, episodes=512, num_gates=12)


def test_variable_oracle_requires_every_counted_gate_and_envelope() -> None:
    metrics = _passing_metrics(11)
    assert variable_oracle_passes(metrics, episodes=512, num_gates=11)
    metrics["env/ordered_gate10_sampled"] = 0.0
    assert not variable_oracle_passes(metrics, episodes=512, num_gates=11)
