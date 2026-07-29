import importlib.util
import sys
from pathlib import Path

import numpy as np
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "shadow_v3391_telemetry_policy.py"
SPEC = importlib.util.spec_from_file_location("shadow_v3391_telemetry_policy", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_live_ned_conversion_matches_native_observation_contract():
    gates, settings = module.load_course(
        ROOT / "config" / "drone_race_vq1_v3391_telemetry.ini"
    )
    position_ned = (-12.0, -0.25, 0.5)
    velocity_ned = (-4.0, 0.5, 1.0)
    last_action = np.asarray([0.3, -0.2, 0.1, 0.0], dtype=np.float32)

    live = module.live_observation(
        position_ned,
        velocity_ned,
        1,
        5.0,
        last_action,
        gates,
        settings,
    )
    expected = module.build_observation(
        -np.asarray(position_ned),
        -np.asarray(velocity_ned),
        1,
        5.0,
        last_action,
        gates,
        settings,
    )

    assert np.array_equal(live, expected)
    assert live[23] == np.float32(1.0 / 6.0)
    assert live[25] == 1.0


def test_policy_action_maps_to_local_ned_velocity_with_expected_signs():
    _gates, settings = module.load_course(
        ROOT / "config" / "drone_race_vq1_v3391_telemetry.ini"
    )

    velocity = module.action_to_ned_velocity(
        np.asarray([0.5, -0.25, 0.75, 1.0], dtype=np.float32), settings
    )

    assert np.allclose(velocity, (-6.0, 2.0, 6.0))


def test_reset_ready_rejects_stale_disarmed_or_far_state():
    status = SimpleNamespace(
        race_start_boot_time_ms=3200,
        active_gate_index=0,
        race_finish_time_ns=-1,
    )
    state = SimpleNamespace(
        race_status=status,
        base_mode=193,
        system_status=4,
        local_position_ned_m=(0.0, 0.0, 0.0),
        local_velocity_ned_m_s=(0.0, 0.0, 0.0),
    )

    assert module.reset_ready_state(state, 5.0)
    state.base_mode = 65
    assert not module.reset_ready_state(state, 5.0)
    state.base_mode = 193
    state.local_position_ned_m = (6.0, 0.0, 0.0)
    assert not module.reset_ready_state(state, 5.0)


def test_track_transfer_compares_base_to_configured_aperture_center():
    gates, _settings = module.load_course(
        ROOT / "config" / "drone_race_vq1_v3391_telemetry.ini"
    )
    transferred = []
    for gate_id, center_ned in enumerate(-gates):
        transferred.append(
            SimpleNamespace(
                gate_id=gate_id,
                position_ned_x=center_ned[0],
                position_ned_y=center_ned[1],
                position_ned_z=center_ned[2] + 1.36,
                height_m=2.72,
            )
        )
    report = module.compare_transferred_gates(
        SimpleNamespace(track_gates=tuple(transferred)), gates
    )
    assert report["available_in_shadow_window"]
    assert report["max_position_error_m"] < 1e-6
