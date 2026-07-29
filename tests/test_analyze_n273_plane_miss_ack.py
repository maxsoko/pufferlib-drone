from pathlib import Path

import pytest

from scripts import analyze_n273_plane_miss_ack as module


ROOT = Path(__file__).resolve().parents[1]
N273 = Path(
    "/mnt/c/Users/anon/code/pufferlib-drone/.n253-shadow/logs/sitl/"
    "n273_v3391_n259_cross9_xgain2_full_lap_001.json"
)


def test_interpolate_x_plane_crossing():
    rows = [
        {"elapsed_s": 1.0, "position_ned_m": [-1.0, 1.0, 2.0], "velocity_ned_m_s": [-2.0, 0.0, 0.0]},
        {"elapsed_s": 2.0, "position_ned_m": [-3.0, 3.0, 4.0], "velocity_ned_m_s": [-4.0, 2.0, 2.0]},
    ]
    crossing = module.interpolate_x_plane_crossing(rows, module.np.asarray([-2.0, 2.0, 3.0]))
    assert crossing["elapsed_s"] == pytest.approx(1.5)
    assert crossing["position_ned_m"] == pytest.approx([-2.0, 2.0, 3.0])
    assert crossing["aperture_yz_error_m"] == pytest.approx(0.0)


def test_real_n273_diagnosis_preregisters_bounded_n275():
    if not N273.exists():
        pytest.skip("N273 Windows evidence is not mounted")
    report = module.build_report(
        N273,
        ROOT / "logs/sitl/n271_n270_three_trace_governor_sweep.json",
        ROOT / "config/drone_race_vq1_v3391_telemetry.ini",
        ROOT / "scripts/run_v3391_telemetry_policy.py",
    )
    diagnosis = report["gate4_diagnosis"]
    assert diagnosis["interpolated_plane_crossing"]["aperture_yz_error_m"] < 0.3
    assert diagnosis["latest_status_minus_crossing_ms"] < 0.0
    assert not diagnosis["latest_status_was_post_crossing"]
    assert report["correction"]["legacy_distance_m_unchanged"] == 1.5
    assert report["correction"]["status_ack_grace_s"] == 0.35
    assert report["preregistered_next_action"]["experiment"] == "N275"
    assert report["preregistered_next_action"]["stop_after_gate_index"] == 4
