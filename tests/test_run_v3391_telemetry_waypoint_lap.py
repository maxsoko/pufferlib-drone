import importlib.util
import math
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_v3391_telemetry_waypoint_lap.py"
)
SPEC = importlib.util.spec_from_file_location("run_v3391_telemetry_waypoint_lap", MODULE_PATH)
lap = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(MODULE_PATH.parent))
sys.modules[SPEC.name] = lap
SPEC.loader.exec_module(lap)


def test_segment_command_tracks_straight_negative_north_segment():
    command = lap.segment_velocity_command(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (-20.0, 0.0, 0.0),
        speed_m_s=5.0,
        cross_track_gain_s_inv=1.0,
        max_cross_track_correction_m_s=2.0,
    )
    assert command.velocity_ned_m_s == pytest.approx((-5.0, 0.0, 0.0))
    assert command.along_to_gate_m == pytest.approx(20.0)
    assert command.cross_track_error_m == pytest.approx(0.0)


def test_segment_command_corrects_cross_track_error_toward_centerline():
    command = lap.segment_velocity_command(
        (-10.0, 2.0, -1.0),
        (0.0, 0.0, 0.0),
        (-20.0, 0.0, 0.0),
        speed_m_s=5.0,
        cross_track_gain_s_inv=1.0,
        max_cross_track_correction_m_s=1.0,
    )
    assert command.velocity_ned_m_s[0] == pytest.approx(-5.0)
    assert command.velocity_ned_m_s[1] < 0.0
    assert command.velocity_ned_m_s[2] > 0.0
    assert math.hypot(command.correction_m_s[1], command.correction_m_s[2]) == pytest.approx(1.0)


def test_segment_command_rejects_degenerate_or_invalid_configuration():
    with pytest.raises(ValueError):
        lap.segment_velocity_command(
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            (1.0, 2.0, 3.0),
            speed_m_s=1.0,
            cross_track_gain_s_inv=1.0,
            max_cross_track_correction_m_s=1.0,
        )
    with pytest.raises(ValueError):
        lap.segment_velocity_command(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            speed_m_s=0.0,
            cross_track_gain_s_inv=1.0,
            max_cross_track_correction_m_s=1.0,
        )


def test_gate_speed_schedule_is_smooth_and_bounded():
    values = [
        lap.gate_scheduled_speed(
            distance,
            cruise_speed_m_s=5.0,
            crossing_speed_m_s=1.5,
            slowdown_distance_m=5.0,
        )
        for distance in (-1.0, 0.0, 1.0, 2.5, 4.0, 5.0, 8.0)
    ]
    assert values[0] == pytest.approx(1.5)
    assert values[1] == pytest.approx(1.5)
    assert values[-2] == pytest.approx(5.0)
    assert values[-1] == pytest.approx(5.0)
    assert values == sorted(values)


def test_finish_requires_all_six_gates_and_nonnegative_official_time():
    valid = lap.RaceStatus(1, 1, 12, 6, 11)
    incomplete = lap.RaceStatus(1, 1, -1, 5, 11)
    delayed = lap.RaceStatus(1, 1, -1, 6, 11)
    assert lap.is_valid_finish(valid, 6)
    assert not lap.is_valid_finish(incomplete, 6)
    assert not lap.is_valid_finish(delayed, 6)


def test_only_centered_low_threat_gate_contact_is_admitted():
    low_gate = {"id": 1001, "threat_level": 1}
    assert lap.is_centered_low_threat_gate_contact(
        low_gate,
        (-23.08, -0.397, -0.026),
        (-23.298, -0.4, -0.032),
        enabled=True,
        radius_m=0.5,
    )
    assert not lap.is_centered_low_threat_gate_contact(
        {"id": 1001, "threat_level": 2},
        (-23.08, -0.397, -0.026),
        (-23.298, -0.4, -0.032),
        enabled=True,
        radius_m=0.5,
    )
    assert not lap.is_centered_low_threat_gate_contact(
        low_gate,
        (-23.08, 0.7, -0.026),
        (-23.298, -0.4, -0.032),
        enabled=True,
        radius_m=0.5,
    )
    assert not lap.is_centered_low_threat_gate_contact(
        low_gate,
        (-23.08, -0.397, -0.026),
        (-23.298, -0.4, -0.032),
        enabled=False,
        radius_m=0.5,
    )


def test_gate_ned_z_offset_is_explicitly_down_positive():
    assert lap.offset_gate_position((-23.0, -0.4, -0.03), 0.4) == pytest.approx(
        (-23.0, -0.4, 0.37)
    )
    assert lap.offset_gate_position((-23.0, -0.4, -0.03), -1.36) == pytest.approx(
        (-23.0, -0.4, -1.39)
    )


def test_source_rate_limits_arm_commands_and_has_vertical_launch_phase():
    source = MODULE_PATH.read_text()
    assert "next_arm_s = now_s + 0.1" in source
    assert "scheduled_start_s = now_s + max(0.0, remaining_from_status_s)" in source
    assert 'mode="body_rates"' in source
    assert '"attitude_launch"' in source
    assert 'launch_support_contact = (' in source
    assert 'collision_event["id"] == 1002' in source
    assert '"launch_climb" if launch_active else "segment_track"' in source
