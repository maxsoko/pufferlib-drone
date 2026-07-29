from pathlib import Path

import pytest

from scripts import analyze_n277_equal_timestamp_ack as module


ROOT = Path(__file__).resolve().parents[1]
N277 = Path(
    "/mnt/c/Users/anon/code/pufferlib-drone/.n253-shadow/logs/sitl/"
    "n277_v3391_n259_cross9_xgain2_ack350_full_lap_confirm_002.json"
)


def test_interpolate_gate_plane():
    rows = [
        {"elapsed_s": 1.0, "position_ned_m": [-1.0, 1.0, 2.0]},
        {"elapsed_s": 2.0, "position_ned_m": [-3.0, 3.0, 4.0]},
    ]
    crossing = module.interpolate_gate_plane(rows, module.np.asarray([-2.0, 2.0, 3.0]))
    assert crossing["elapsed_s"] == pytest.approx(1.5)
    assert crossing["position_ned_m"] == pytest.approx([-2.0, 2.0, 3.0])
    assert crossing["absolute_y_error_m"] == pytest.approx(0.0)
    assert crossing["absolute_z_error_m"] == pytest.approx(0.0)


def test_real_n277_equal_timestamp_diagnosis():
    if not N277.exists():
        pytest.skip("N277 Windows evidence is not mounted")
    report = module.build_report(
        N277,
        ROOT / "logs/sitl/n274_n273_plane_miss_ack_diagnosis.json",
        ROOT / "config/drone_race_vq1_v3391_telemetry.ini",
        ROOT / "scripts/run_v3391_telemetry_policy.py",
    )
    diagnosis = report["equal_timestamp_diagnosis"]
    assert diagnosis["crossing_boot_time_ms"] == diagnosis["race_status_boot_time_ms"]
    assert not diagnosis["strictly_post_crossing_status_observed"]
    assert diagnosis["elapsed_since_crossing_ms"] < diagnosis["ack_grace_ms"]
    assert diagnosis["vehicle_center_inside_aperture"]
    assert report["correction"]["distance_and_timeout_guards_unchanged"]
