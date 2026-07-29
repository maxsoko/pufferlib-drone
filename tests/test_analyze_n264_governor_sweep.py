import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_n264_governor_sweep",
    ROOT / "scripts" / "analyze_n264_governor_sweep.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_scheduled_speed_preserves_endpoints_and_smooth_midpoint():
    assert MODULE.scheduled_speed(
        0.0,
        cruise_speed_m_s=8.0,
        crossing_speed_m_s=4.0,
        slowdown_distance_m=10.0,
    ) == pytest.approx(4.0)
    assert MODULE.scheduled_speed(
        10.0,
        cruise_speed_m_s=8.0,
        crossing_speed_m_s=4.0,
        slowdown_distance_m=10.0,
    ) == pytest.approx(8.0)
    assert MODULE.scheduled_speed(
        5.0,
        cruise_speed_m_s=8.0,
        crossing_speed_m_s=4.0,
        slowdown_distance_m=10.0,
    ) == pytest.approx(6.0)


def test_fit_response_axes_recovers_first_order_coefficients():
    trace = []
    velocity = np.asarray([0.0, 0.0, 0.0])
    command = np.asarray([4.0, -2.0, 1.0])
    dt = 0.1
    command_gain = np.asarray([1.0, 2.0, 3.0])
    velocity_decay = np.asarray([0.5, 1.0, 1.5])
    for index in range(40):
        trace.append(
            {
                "elapsed_s": index * dt,
                "active_gate_index": 0,
                "command_velocity_ned_m_s": command.tolist(),
                "velocity_ned_m_s": velocity.tolist(),
            }
        )
        velocity = velocity + dt * (command_gain * command - velocity_decay * velocity)
    fitted = MODULE.fit_response_axes(trace)
    for axis, result in enumerate(fitted):
        assert result.command_gain_s_inv == pytest.approx(command_gain[axis])
        assert result.velocity_decay_s_inv == pytest.approx(velocity_decay[axis])
        assert result.steady_state_gain == pytest.approx(
            command_gain[axis] / velocity_decay[axis]
        )
        assert result.rmse_m_s2 < 1e-10


def test_real_n264_replay_and_reduced_sweep_retain_baseline():
    import json

    source = ROOT / "logs" / "sitl" / "n264_v3391_n259_governed_full_lap_001.json"
    report = json.loads(source.read_text())
    gates, _ = MODULE.load_ned_gate_centers(
        ROOT / "config" / "drone_race_vq1_v3391_telemetry.ini"
    )
    response = MODULE.fit_response_axes(report["trace"])
    baseline = MODULE.GovernorParameters(**MODULE.BASELINE_PARAMETERS)
    replay = MODULE.replay_governor(
        report["trace"], gates, response, (baseline,) * 6
    )
    assert len(replay) == 6
    official = MODULE.official_gate_times(report)
    assert abs(replay[-1].raw_crossing_time_s - official[-1]) < 0.5
    sweep = MODULE.sweep_parameters(
        report["trace"],
        gates,
        response,
        official,
        replay,
        baseline,
        crossing_speeds=(4.0, 7.5),
        maximum_along_speeds=(8.0,),
        slowdown_distances=(10.0,),
        cross_track_gains=(1.5,),
        correction_caps=(4.0,),
    )
    by_crossing = {
        item["parameters"]["crossing_speed_m_s"]: item for item in sweep
    }
    assert by_crossing[4.0]["predicted_finish_time_s"] == pytest.approx(
        official[-1]
    )
    assert by_crossing[7.5]["safe_offline"]
    assert by_crossing[7.5]["predicted_finish_time_s"] < 27.0
    assert by_crossing[7.5]["parameter_change_count"] == 1


def test_minimal_target_selection_chooses_slowest_equal_change_rung():
    candidates = [
        {
            "safe_offline": True,
            "predicted_finish_time_s": 26.1,
            "parameter_change_count": 1,
            "maximum_crossing_error_m": 0.3,
            "parameters": {"crossing_speed_m_s": 8.0},
        },
        {
            "safe_offline": True,
            "predicted_finish_time_s": 26.8,
            "parameter_change_count": 1,
            "maximum_crossing_error_m": 0.2,
            "parameters": {"crossing_speed_m_s": 7.5},
        },
    ]
    chosen = MODULE.choose_minimal_under_target(candidates, 27.0)
    assert chosen["parameters"]["crossing_speed_m_s"] == 7.5
