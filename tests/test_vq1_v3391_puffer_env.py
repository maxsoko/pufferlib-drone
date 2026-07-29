from __future__ import annotations

import configparser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "drone_race_vq1_v3391_telemetry.ini"

NED_GATES = (
    (-23.2979679108, -0.3999023438, -1.3919580063),
    (-46.8937492371, -2.4999902248, 3.7080418015),
    (-74.5937500000, 1.2000097036, 12.3080412292),
    (-111.4937438965, -5.0999898911, 23.2080408478),
    (-135.4937438965, -0.7999901772, 23.9956537628),
    (-159.1937408447, -4.3999900818, 24.6080404663),
)


def _config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    assert config.read(CONFIG) == [str(CONFIG)]
    return config


def test_v3391_profile_contains_six_live_proven_aperture_centers() -> None:
    config = _config()
    env = config["env"]
    assert env.getint("num_gates") == 6
    assert env.getint("use_custom_gate_layout") == 1
    assert env.getint("start_gate_index") == 0
    assert env.getint("mixed_start_curriculum") == 0
    for gate_index, ned in enumerate(NED_GATES):
        native = tuple(
            env.getfloat(f"gate{gate_index}_{axis}") for axis in ("x", "y", "z")
        )
        assert native == tuple(-value for value in ned)


def test_v3391_profile_uses_deployable_telemetry_velocity_contract() -> None:
    config = _config()
    env = config["env"]
    assert env.getint("interface_mode") == 3
    assert env.getfloat("max_cmd_forward") == 12.0
    assert env.getfloat("telemetry_velocity_response_tau_s") > 0.0
    assert env.getfloat("telemetry_velocity_max_accel_m_s2") > 0.0
    assert env.getfloat("telemetry_velocity_teacher_speed_m_s") > 0.0
    assert env.getint("observable_gate_progress") == 1
    assert env.getint("observable_gate_phase_onehot") == 1
    assert 0.0 < env.getfloat("gate_radius") < 0.75
