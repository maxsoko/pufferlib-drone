import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "run_v3391_telemetry_policy.py"
SPEC = importlib.util.spec_from_file_location("run_v3391_telemetry_policy", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_live_policy_runner_rejects_out_of_contract_command_rate(tmp_path):
    args = type("Args", (), {"duration_s": 1.0, "heartbeat_hz": 2.0, "command_hz": 100.0})()
    with pytest.raises(ValueError, match="command_hz"):
        module.run_policy(args)


def test_gate_governor_keeps_policy_along_speed_and_corrects_cross_track():
    governed = module.gate_governed_velocity(
        position_ned_m=(5.0, 2.0, -3.0),
        segment_start_ned_m=(0.0, 0.0, 0.0),
        gate_center_ned_m=(10.0, 0.0, 0.0),
        policy_velocity_ned_m_s=(7.0, 3.0, -2.0),
        crossing_speed_m_s=4.0,
        maximum_along_speed_m_s=8.0,
        slowdown_distance_m=10.0,
        cross_track_gain_s_inv=1.5,
        maximum_cross_track_correction_m_s=4.0,
    )

    assert governed.policy_along_speed_m_s == pytest.approx(7.0)
    assert governed.scheduled_along_speed_m_s == pytest.approx(5.5)
    assert governed.along_to_gate_m == pytest.approx(5.0)
    assert governed.cross_track_error_m == pytest.approx(13**0.5)
    assert governed.velocity_ned_m_s[0] == pytest.approx(5.5)
    assert governed.velocity_ned_m_s[1] < 0.0
    assert governed.velocity_ned_m_s[2] > 0.0


def test_gate_governor_floors_a_reversing_policy_at_crossing_speed():
    governed = module.gate_governed_velocity(
        position_ned_m=(9.0, 0.2, 0.1),
        segment_start_ned_m=(0.0, 0.0, 0.0),
        gate_center_ned_m=(10.0, 0.0, 0.0),
        policy_velocity_ned_m_s=(-2.0, 0.0, 0.0),
        crossing_speed_m_s=4.0,
        maximum_along_speed_m_s=8.0,
        slowdown_distance_m=10.0,
        cross_track_gain_s_inv=1.5,
        maximum_cross_track_correction_m_s=4.0,
    )

    assert governed.policy_along_speed_m_s == pytest.approx(-2.0)
    assert governed.scheduled_along_speed_m_s == pytest.approx(4.0)
    assert governed.velocity_ned_m_s[0] == pytest.approx(4.0)


def test_plane_miss_guard_preserves_legacy_abort_by_default():
    decision = module.plane_miss_guard_decision(
        along_to_gate_m=-1.6,
        abort_distance_m=1.5,
        crossing_boot_time_ms=1000,
        current_boot_time_ms=1200,
        race_status_boot_time_ms=990,
        status_ack_grace_s=0.0,
    )
    assert decision.should_abort
    assert decision.reason == "legacy_distance_abort"


def test_plane_miss_guard_defers_until_post_crossing_status_ack():
    decision = module.plane_miss_guard_decision(
        along_to_gate_m=-1.6,
        abort_distance_m=1.5,
        crossing_boot_time_ms=1000,
        current_boot_time_ms=1200,
        race_status_boot_time_ms=990,
        status_ack_grace_s=0.35,
    )
    assert not decision.should_abort
    assert decision.deferred_for_status_ack
    assert decision.reason == "awaiting_status_ack"


def test_plane_miss_guard_aborts_after_same_gate_ack():
    decision = module.plane_miss_guard_decision(
        along_to_gate_m=-1.6,
        abort_distance_m=1.5,
        crossing_boot_time_ms=1000,
        current_boot_time_ms=1200,
        race_status_boot_time_ms=1001,
        status_ack_grace_s=0.35,
    )
    assert decision.should_abort
    assert decision.reason == "post_crossing_status_still_same_gate"


def test_plane_miss_guard_does_not_treat_equal_millisecond_as_post_crossing():
    decision = module.plane_miss_guard_decision(
        along_to_gate_m=-1.6,
        abort_distance_m=1.5,
        crossing_boot_time_ms=1000,
        current_boot_time_ms=1211,
        race_status_boot_time_ms=1000,
        status_ack_grace_s=0.35,
    )
    assert not decision.should_abort
    assert decision.deferred_for_status_ack
    assert decision.reason == "awaiting_status_ack"


def test_plane_miss_guard_aborts_on_bounded_ack_timeout():
    decision = module.plane_miss_guard_decision(
        along_to_gate_m=-1.6,
        abort_distance_m=1.5,
        crossing_boot_time_ms=1000,
        current_boot_time_ms=1350,
        race_status_boot_time_ms=990,
        status_ack_grace_s=0.35,
    )
    assert decision.should_abort
    assert decision.reason == "status_ack_timeout"
