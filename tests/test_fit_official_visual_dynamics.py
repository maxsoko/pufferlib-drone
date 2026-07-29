from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fit_official_visual_dynamics as fit


def _sample(index: int, position: np.ndarray) -> fit.TraceSample:
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    return fit.TraceSample(
        report="synthetic.json",
        report_fold=0,
        elapsed_s=0.1 * index,
        gate_index=3,
        body_ned_m=np.asarray([position[0], position[1], -position[2]]),
        relative_world_m=position,
        quaternion_wxyz=quaternion,
        euler_rpy_rad=np.zeros(3),
        gyro_rpy_rad_s=np.zeros(3),
        normalized_action=np.zeros(4),
    )


def test_derivative_rows_recover_smooth_raw_aperture_motion() -> None:
    segment = []
    for index in range(13):
        time_s = 0.1 * index
        position = np.asarray(
            [30.0 - 5.0 * time_s - 0.5 * time_s**2, 2.0, -3.0]
        )
        segment.append(_sample(index, position))
    rows, lookup, quality = fit._derivative_rows(
        [segment],
        window=9,
        maximum_local_fit_rmse_m=0.01,
        maximum_observed_step_speed_m_s=30.0,
        maximum_acceleration_norm_m_s2=30.0,
    )
    assert len(rows) == 5
    assert len(lookup) == 5
    assert quality["rejected_local_fit_rows"] == 0
    np.testing.assert_allclose(rows[2].velocity_world_m_s, [5.6, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(rows[2].acceleration_world_m_s2, [1.0, 0.0, 0.0], atol=1e-8)


def test_derivative_rows_reject_camera_alias_jump() -> None:
    segment = [
        _sample(index, np.asarray([30.0 - index * 0.5, 0.0, 0.0]))
        for index in range(13)
    ]
    segment[6] = _sample(6, np.asarray([70.0, 20.0, 0.0]))
    rows, _lookup, quality = fit._derivative_rows(
        [segment],
        window=9,
        maximum_local_fit_rmse_m=0.75,
        maximum_observed_step_speed_m_s=30.0,
        maximum_acceleration_norm_m_s2=30.0,
    )
    assert not rows
    assert quality["rejected_local_fit_rows"] > 0


def test_forward_fit_has_zero_bias_and_nonnegative_damping() -> None:
    rows = []
    for index in range(20):
        pitch = -0.20 + 0.02 * index
        quaternion = fit._euler_to_quaternion((0.0, pitch, 0.0))
        up = fit._quaternion_rotate(quaternion, (0.0, 0.0, 1.0))
        thrust_feature = -up[0] * fit.HOVER_THRUST
        velocity = np.asarray([2.0 + 0.2 * index, 0.0, 0.0])
        acceleration = np.asarray([60.0 * thrust_feature - 0.1 * velocity[0], 0.0, 0.0])
        rows.append(
            fit.DerivativeRow(
                report=f"r{index}.json",
                report_fold=index % 5,
                segment_index=index,
                sample_index=4,
                elapsed_s=1.0,
                position_world_m=np.zeros(3),
                velocity_world_m_s=velocity,
                acceleration_world_m_s2=acceleration,
                quaternion_wxyz=quaternion,
                normalized_action=np.zeros(4),
                local_position_fit_rmse_m=0.0,
                maximum_observed_step_speed_m_s=5.0,
            )
        )
    model = fit._fit_translation(rows)
    forward = model["axes"][0]
    assert forward["forward_zero_bias_constraint"]
    assert forward["bias_m_s2"] == 0.0
    assert forward["thrust_gain"] >= 0.0
    assert forward["linear_drag_per_s"] >= 0.0

