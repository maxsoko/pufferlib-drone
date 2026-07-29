from __future__ import annotations

import pytest

from scripts.eval_vq2_native_oracle import (
    BACKEND_ENV_NAME,
    ENV_NAME,
    EPISODE_STEPS,
    fixed_environment,
    fixed_overrides,
    flatten_log,
    load_fixed_config,
    native_step_budget,
    oracle_passes,
)


class FakePufferl:
    @staticmethod
    def load_config(env_name: str) -> dict:
        assert env_name == ENV_NAME
        import sys

        pairs = dict(zip(sys.argv[1::2], sys.argv[2::2], strict=True))
        return {
            "backend_env_name": BACKEND_ENV_NAME,
            "vec": {"total_agents": int(pairs["--vec.total-agents"])},
            "env": {"start_gate_index": 0},
        }


def _pairs(values: list[str]) -> dict[str, str]:
    return dict(zip(values[::2], values[1::2], strict=True))


def test_fixed_oracle_contract_is_full_course_true_aperture_and_full_teacher() -> None:
    pairs = _pairs(fixed_overrides(agents=64, seed=42002))
    assert pairs["--vec.total-agents"] == "64"
    environment = fixed_environment()
    assert environment["num_gates"] == 6
    assert environment["gate_radius"] == 0.75
    assert environment["gate_radius_randomize"] == 0
    assert environment["gate_radius_profile_mix"] == 0
    assert environment["gate_position_domain_randomize"] == 0
    assert environment["course_geometry_scale_randomize"] == 0
    assert environment["sitl_plant_domain_randomize"] == 0
    assert environment["use_custom_start"] == 0
    assert environment["teacher_action_blend"] == 1.0
    assert environment["teacher_course_spline"] == 1
    assert environment["teacher_pitch_speed_target_m_s"] == 4.0
    assert environment["teacher_roll_until_gate_index"] == 6
    assert environment["teacher_yaw_control"] == 1
    assert EPISODE_STEPS == 2560


def test_config_assigns_one_exact_full_start_episode_per_agent() -> None:
    config, _overrides = load_fixed_config(
        FakePufferl, agents=64, episodes=64, seed=42002
    )
    assert config["env"]["evaluation_episode_limit"] == 1
    assert config["env"]["evaluation_episode_offset"] == 0
    assert config["env"]["gate_local_start_curriculum"] == 0
    assert config["env"]["gate_local_start_probability"] == 0.0
    assert config["env"]["mixed_start_curriculum"] == 0
    assert config["env"]["segment_start_probability"] == 0.0


def test_config_rejects_partial_agent_episode_allocation() -> None:
    with pytest.raises(ValueError, match="multiple of agents"):
        load_fixed_config(FakePufferl, agents=64, episodes=65, seed=42002)


def test_native_step_budget_covers_every_assigned_episode_boundary() -> None:
    assert native_step_budget(max_steps=11520, episodes=4096, agents=512) == (
        11521 * 8
    )
    with pytest.raises(ValueError, match="invalid"):
        native_step_budget(max_steps=11520, episodes=513, agents=512)


def test_direct_controller_changes_only_teacher_lateral_vertical_law() -> None:
    spline = fixed_environment("spline")
    direct = fixed_environment("direct")
    assert spline["teacher_course_spline"] == 1
    assert direct["teacher_course_spline"] == 0
    assert direct["teacher_roll_per_m"] == 0.35
    assert direct["teacher_roll_rate_per_m_s"] == 0.10
    assert direct["teacher_thrust_per_m"] == 0.30
    assert direct["teacher_thrust_rate_per_m_s"] == 0.10
    assert direct["teacher_thrust_world_frame"] == 1
    for key in spline:
        if key != "teacher_course_spline":
            assert direct[key] == spline[key]


def test_segment_controller_is_one_default_off_reference_switch() -> None:
    spline = fixed_environment("spline")
    segment = fixed_environment("segment")
    assert spline["teacher_course_spline"] == 1
    assert spline["teacher_segment_minimum_jerk"] == 0
    assert segment["teacher_course_spline"] == 0
    assert segment["teacher_segment_minimum_jerk"] == 1
    changed = {key for key in spline if spline[key] != segment[key]}
    assert changed == {
        "teacher_course_spline",
        "teacher_segment_minimum_jerk",
    }


def test_alignment_governor_is_explicitly_default_off() -> None:
    direct = fixed_environment("direct")
    governed = fixed_environment("governed")
    assert direct["teacher_alignment_governor"] == 0
    assert governed["teacher_alignment_governor"] == 1
    assert governed["teacher_course_spline"] == 0
    assert governed["teacher_segment_minimum_jerk"] == 0


def test_randomized_course_contract_is_explicit_and_keeps_true_aperture() -> None:
    randomized = fixed_environment("governed", randomized_course=True)
    assert randomized["gate_radius"] == 0.75
    assert randomized["gate_radius_randomize"] == 0
    assert randomized["gate_position_domain_randomize"] == 1
    assert randomized["gate_position_randomize_from_index"] == 0
    assert randomized["gate_position_jitter_x"] == 3.0
    assert randomized["gate_position_jitter_y"] == 5.0
    assert randomized["gate_position_jitter_z"] == 0.5
    assert randomized["course_geometry_scale_randomize"] == 1
    assert randomized["course_geometry_scale_min"] == 0.35
    assert randomized["course_geometry_scale_max"] == 1.0


def test_slow_solve_first_variant_changes_only_speed_and_feasible_horizon() -> None:
    fast = fixed_environment("direct")
    slow = fixed_environment(
        "direct", target_speed_m_s=2.5, episode_seconds=60.0
    )
    assert slow["teacher_pitch_speed_target_m_s"] == 2.5
    assert slow["time_limit_seconds"] == 60.0
    assert slow["max_steps"] == 3840
    changed = {
        key for key in fast if fast[key] != slow[key]
    }
    assert changed == {
        "teacher_pitch_speed_target_m_s",
        "time_limit_seconds",
        "max_steps",
    }


def test_direct_pd_gains_are_explicit_and_nonnegative() -> None:
    tuned = fixed_environment(
        "direct", roll_per_m=0.35, roll_rate_per_m_s=0.7
    )
    assert tuned["teacher_roll_per_m"] == 0.35
    assert tuned["teacher_roll_rate_per_m_s"] == 0.7
    with pytest.raises(ValueError, match="positive"):
        fixed_environment("direct", roll_rate_per_m_s=-0.1)


def test_true_velocity_damping_is_explicitly_default_off() -> None:
    historical = fixed_environment("direct")
    damped = fixed_environment("direct", true_velocity_damping=True)
    assert historical["teacher_true_velocity_damping"] == 0
    assert damped["teacher_true_velocity_damping"] == 1
    changed = {key for key in historical if historical[key] != damped[key]}
    assert changed == {"teacher_true_velocity_damping"}


def test_raw_vector_log_is_normalized_to_env_namespace() -> None:
    class LogPufferl:
        @staticmethod
        def unroll_nested_dict(value):
            return value.items()

    assert flatten_log(LogPufferl, {"n": 64, "env/crash": 0}) == {
        "env/n": 64.0,
        "env/crash": 0.0,
    }


def test_oracle_decision_rule_requires_every_gate() -> None:
    passing = {
        "env/n": 64.0,
        "env/success_rate": 1.0,
        "env/gates_passed": 6.0,
        "env/crash": 0.0,
        "env/timeout": 0.0,
        "env/missed_gate": 0.0,
        "env/out_of_order": 0.0,
        "env/valid_run_rate": 1.0,
        "env/crossing_margin_violation": 0.0,
        "env/action_envelope_violation": 0.0,
        "env/wire_rate_envelope_violation": 0.0,
        "env/thrust_envelope_violation": 0.0,
    }
    assert oracle_passes(passing, episodes=64)
    for key in passing:
        failed = dict(passing)
        failed[key] = 0.0 if passing[key] else 1.0
        assert not oracle_passes(failed, episodes=64), key
