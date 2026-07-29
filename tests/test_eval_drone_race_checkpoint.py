import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_drone_race_checkpoint.py"
SCRIPTS = MODULE_PATH.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("eval_drone_race_checkpoint", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_validate_backend_env_accepts_exact_env_match():
    module._validate_backend_env("drone_race_competition", "drone_race_competition", "drone_race")


def test_validate_backend_env_accepts_backend_match():
    module._validate_backend_env("drone_race", "drone_race_competition", "drone_race")


def test_validate_backend_env_rejects_mismatch():
    with pytest.raises(RuntimeError, match="backend mismatch"):
        module._validate_backend_env("drone", "drone_race_competition", "drone_race")


def test_configure_exact_episode_limit_assigns_equal_quota():
    cfg = {"vec": {"total_agents": 1024}, "env": {"num_drones": 1}}
    assert module._configure_exact_episode_limit(cfg, 4096, 2) == 4
    assert cfg["env"]["evaluation_episode_limit"] == 4
    assert cfg["env"]["evaluation_episode_offset"] == 2


def test_configure_exact_episode_limit_rejects_nonmultiple():
    cfg = {"vec": {"total_agents": 1024}, "env": {"num_drones": 1}}
    with pytest.raises(ValueError, match="multiple"):
        module._configure_exact_episode_limit(cfg, 4095)


def test_configure_exact_episode_limit_rejects_multi_agent_env():
    cfg = {"vec": {"total_agents": 1024}, "env": {"num_drones": 2}}
    with pytest.raises(ValueError, match="num_drones"):
        module._configure_exact_episode_limit(cfg, 4096)


def test_configure_exact_episode_limit_rejects_negative_offset():
    cfg = {"vec": {"total_agents": 1024}, "env": {"num_drones": 1}}
    with pytest.raises(ValueError, match="offset"):
        module._configure_exact_episode_limit(cfg, 4096, -1)


def test_derived_metrics_condition_gate_exit_velocity_on_sampled_transitions():
    metrics = {
        "env/gate0_exit_sampled": 0.25,
        "env/gate0_exit_vx": 5.0,
        "env/gate0_exit_vy": 0.25,
        "env/gate0_exit_vz": -0.5,
    }
    module._add_derived_metrics(metrics)
    assert metrics["env/avg_gate0_exit_vx"] == pytest.approx(20.0)
    assert metrics["env/avg_gate0_exit_vy"] == pytest.approx(1.0)
    assert metrics["env/avg_gate0_exit_vz"] == pytest.approx(-2.0)


def test_derived_metrics_condition_gate2_crossing_and_exit_on_successful_passes():
    metrics = {
        "env/gate2_crossing_sampled": 0.5,
        "env/gate2_crossing_radial": 0.25,
        "env/gate2_crossing_right": -0.15,
        "env/gate2_crossing_vertical": 0.20,
        "env/gate2_exit_vx": 10.0,
        "env/gate2_exit_vy": 1.5,
        "env/gate2_exit_vz": -2.5,
    }
    module._add_derived_metrics(metrics)
    assert metrics["env/avg_gate2_crossing_radial"] == pytest.approx(0.5)
    assert metrics["env/avg_gate2_crossing_right"] == pytest.approx(-0.3)
    assert metrics["env/avg_gate2_crossing_vertical"] == pytest.approx(0.4)
    assert metrics["env/avg_gate2_exit_vx"] == pytest.approx(20.0)
    assert metrics["env/avg_gate2_exit_vy"] == pytest.approx(3.0)
    assert metrics["env/avg_gate2_exit_vz"] == pytest.approx(-5.0)


def test_derived_metrics_partition_gate2_transitions_by_course_outcome():
    metrics = {
        "env/gate2_completion_sampled": 0.25,
        "env/gate2_completion_crossing_radial": 0.10,
        "env/gate2_completion_crossing_right": -0.05,
        "env/gate2_completion_crossing_vertical": 0.075,
        "env/gate2_completion_exit_vx": 5.0,
        "env/gate2_completion_exit_vy": 0.5,
        "env/gate2_completion_exit_vz": -1.0,
        "env/gate2_downstream_failure_sampled": 0.5,
        "env/gate2_downstream_failure_crossing_radial": 0.3,
        "env/gate2_downstream_failure_crossing_right": 0.2,
        "env/gate2_downstream_failure_crossing_vertical": -0.1,
        "env/gate2_downstream_failure_exit_vx": 10.0,
        "env/gate2_downstream_failure_exit_vy": 0.75,
        "env/gate2_downstream_failure_exit_vz": -2.5,
    }
    module._add_derived_metrics(metrics)
    assert metrics["env/avg_gate2_completion_crossing_radial"] == pytest.approx(0.4)
    assert metrics["env/avg_gate2_completion_exit_vy"] == pytest.approx(2.0)
    assert metrics["env/avg_gate2_downstream_failure_crossing_radial"] == pytest.approx(0.6)
    assert metrics["env/avg_gate2_downstream_failure_exit_vy"] == pytest.approx(1.5)


def test_derived_metrics_average_gate3_actions_over_active_steps():
    metrics = {
        "env/gate0_action_steps": 2.0,
        "env/gate0_action_pitch": -1.5,
        "env/gate1_action_steps": 5.0,
        "env/gate1_action_pitch": -1.0,
        "env/gate1_action_roll": 2.0,
        "env/gate1_action_thrust": 0.5,
        "env/gate1_action_yaw": -0.25,
        "env/gate1_early_action_steps": 2.0,
        "env/gate1_early_action_pitch": -0.6,
        "env/gate1_early_action_roll": 1.0,
        "env/gate1_early_action_thrust": 0.4,
        "env/gate1_early_action_yaw": -0.2,
        "env/gate3_action_steps": 4.0,
        "env/gate3_action_pitch": -2.0,
        "env/gate3_action_roll": 1.0,
        "env/gate3_action_thrust": -0.8,
        "env/gate3_action_yaw": 0.4,
    }
    module._add_derived_metrics(metrics)
    assert metrics["env/avg_gate0_action_pitch"] == pytest.approx(-0.75)
    assert metrics["env/avg_gate1_action_pitch"] == pytest.approx(-0.2)
    assert metrics["env/avg_gate1_action_roll"] == pytest.approx(0.4)
    assert metrics["env/avg_gate1_action_thrust"] == pytest.approx(0.1)
    assert metrics["env/avg_gate1_action_yaw"] == pytest.approx(-0.05)
    assert metrics["env/avg_gate1_early_action_pitch"] == pytest.approx(-0.3)
    assert metrics["env/avg_gate1_early_action_roll"] == pytest.approx(0.5)
    assert metrics["env/avg_gate1_early_action_thrust"] == pytest.approx(0.2)
    assert metrics["env/avg_gate1_early_action_yaw"] == pytest.approx(-0.1)
    assert metrics["env/avg_gate3_action_pitch"] == pytest.approx(-0.5)
    assert metrics["env/avg_gate3_action_roll"] == pytest.approx(0.25)
    assert metrics["env/avg_gate3_action_thrust"] == pytest.approx(-0.2)
    assert metrics["env/avg_gate3_action_yaw"] == pytest.approx(0.1)


def test_derived_metrics_report_per_episode_slot_success():
    metrics = {
        "env/episode_slot0_n": 0.25,
        "env/episode_slot0_success": 0.20,
    }
    module._add_derived_metrics(metrics)
    assert metrics["env/episode_slot0_success_rate"] == pytest.approx(0.8)
    assert metrics["env/episode_slot1_success_rate"] == 0.0
