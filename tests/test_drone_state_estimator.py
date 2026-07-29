import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest

from drone_state_estimator import (
    GRAVITY_M_S2,
    AttitudeLoopConfig,
    DeadReckoningEstimator,
    EstimatorConfig,
    attitude_body_rate_command,
    rotate_body_to_world,
    rotate_world_to_body,
)


REST_ACCEL = (-3.0, 0.0, -9.34)  # measured v3379 rest vector (tilted IMU mount)


def calibrated_estimator(**config_kwargs) -> DeadReckoningEstimator:
    est = DeadReckoningEstimator(EstimatorConfig(**config_kwargs))
    for _ in range(25):
        est.add_calibration_sample(REST_ACCEL)
    est.finish_calibration()
    return est


def test_rotation_round_trip():
    vec = (1.0, -2.0, 3.0)
    angles = (0.2, -0.35, 1.1)
    world = rotate_body_to_world(vec, *angles)
    back = rotate_world_to_body(world, *angles)
    assert all(abs(a - b) < 1e-9 for a, b in zip(vec, back))


def test_calibration_recovers_mount_tilt():
    est = calibrated_estimator()
    corrected = est._mount_correct(REST_ACCEL)
    assert abs(corrected[0]) < 1e-6
    assert abs(corrected[1]) < 1e-6
    assert corrected[2] < -9.0


def test_calibration_rejects_spawn_transients():
    est = DeadReckoningEstimator()
    # Values observed live right after MAVLINK_CMD_SIM_RESET (spawn drop).
    assert est.add_calibration_sample((100.9, 0.2, -550.5)) is False
    assert est.calibration_samples == 0
    assert est.add_calibration_sample(REST_ACCEL) is True


def test_collision_spikes_are_counted_not_integrated():
    est = calibrated_estimator(velocity_leak_per_s=0.0)
    est.update_imu(0.0, REST_ACCEL)
    est.update_imu(0.01, (200.0, 0.0, -400.0))  # collision spike
    est.update_imu(0.02, (180.0, 0.0, -380.0))  # same impact, debounced
    est.update_imu(1.0, (200.0, 0.0, -400.0))   # second impact
    assert est.state.collision_events == 2
    # Accel is never integrated: velocity stays exactly zero.
    assert est.state.velocity_ned_m_s == (0.0, 0.0, 0.0)


def test_accel_carries_no_velocity_information():
    # v3379's accelerometer reads the rest vector all flight; a plausible
    # "forward thrust" reading must not create velocity either.
    est = calibrated_estimator()
    for i in range(200):
        est.update_imu(i * 0.01, (2.0, 0.0, -9.6))
    assert est.state.velocity_ned_m_s == (0.0, 0.0, 0.0)
    assert est.state.position_ned_m == (0.0, 0.0, 0.0)


def test_calibration_requires_min_samples():
    est = DeadReckoningEstimator()
    est.add_calibration_sample(REST_ACCEL)
    with pytest.raises(ValueError):
        est.finish_calibration()


def test_stationary_hover_does_not_drift():
    est = calibrated_estimator(velocity_leak_per_s=0.0)
    est.set_commanded_attitude(0.0, 0.0, 0.0)
    for i in range(500):
        est.update_imu(i * 0.01, REST_ACCEL)
    px, py, pz = est.state.position_ned_m
    assert math.sqrt(px * px + py * py + pz * pz) < 0.05


def test_fix_differencing_measures_velocity():
    est = calibrated_estimator(position_correction_gain=1.0, fix_velocity_gain=1.0)
    # Map the gate at 10 m dead ahead, then close at 2 m/s: each fix shows the
    # gate 2 m/30 Hz nearer.
    est.observe_gate(0, (10.0, 0.0, 0.0), time_s=0.0)
    for i in range(1, 31):
        t = i / 30.0
        est.observe_gate(0, (10.0 - 2.0 * t, 0.0, 0.0), time_s=t)
    vx, vy, vz = est.state.velocity_ned_m_s
    assert vx == pytest.approx(2.0, abs=0.15)
    assert abs(vy) < 0.05 and abs(vz) < 0.05
    assert est.state.position_ned_m[0] == pytest.approx(2.0, abs=0.15)


def test_velocity_coasts_and_leaks_between_fixes():
    est = calibrated_estimator(velocity_leak_per_s=0.3)
    est.state.velocity_ned_m_s = (3.0, 0.0, 0.0)
    for i in range(101):  # 1 s of blind coasting
        est.update_imu(i * 0.01, REST_ACCEL)
    vx, _, _ = est.state.velocity_ned_m_s
    # Leak decays velocity but position still advances along the coast.
    assert 1.5 < vx < 3.0
    assert 1.5 < est.state.position_ned_m[0] < 3.0


def test_first_gate_observation_maps_landmark():
    est = calibrated_estimator()
    est.set_commanded_attitude(0.0, 0.0, 0.0)
    pos = est.observe_gate(0, (10.0, 0.0, -2.0))
    assert pos == pytest.approx((10.0, 0.0, -2.0))
    assert est.landmarks[0] == pytest.approx((10.0, 0.0, -2.0))


def test_repeat_observation_corrects_position_drift():
    est = calibrated_estimator(position_correction_gain=0.5)
    est.set_commanded_attitude(0.0, 0.0, 0.0)
    est.observe_gate(0, (10.0, 0.0, 0.0))
    # Inject 2 m of drift, then re-observe the gate at the true relative range.
    est.state.position_ned_m = (2.0, 0.0, 0.0)
    est.observe_gate(0, (10.0, 0.0, 0.0))
    # Fix says we are at ~0; half-gain correction pulls 2.0 -> ~1.0.
    assert est.state.position_ned_m[0] < 1.5
    assert est.state.position_corrections == 1


def test_gate_observation_respects_yaw():
    est = calibrated_estimator()
    est.set_commanded_attitude(0.0, 0.0, math.pi / 2.0)
    pos = est.observe_gate(0, (10.0, 0.0, 0.0))
    # Body-forward 10 m while facing east maps the gate 10 m east.
    assert pos == pytest.approx((0.0, 10.0, 0.0), abs=1e-6)


def test_ahrs_gyro_integrates_pitch():
    est = calibrated_estimator(velocity_leak_per_s=0.0)
    # Rotate pitch-down at 0.5 rad/s for 1 s. The v3379 gyro reports in the
    # command/body frame directly (verified live: pure yaw -> pure zgyro), so
    # no mount correction applies.
    gyro_body = (0.0, -0.5, 0.0)
    # Freefall-magnitude accel keeps the accel-blend out of the loop.
    f_imu = (0.0, 0.0, 0.0)
    for i in range(101):
        est.update_imu(i * 0.01, f_imu, gyro_body)
    assert est.state.pitch == pytest.approx(-0.5, abs=0.02)
    # Commanded attitude must no longer override the gyro-tracked state.
    est.set_commanded_attitude(0.0, 0.3, 0.0)
    assert est.state.pitch == pytest.approx(-0.5, abs=0.02)


def test_body_rate_loop_compensates_measured_plant_sign_and_gain():
    cfg = AttitudeLoopConfig(
        kp_roll=2.0,
        kp_pitch=2.0,
        kp_yaw=2.0,
        body_rate_gain_roll=-2.0,
        body_rate_gain_pitch=-2.0,
        body_rate_gain_yaw=-2.0,
        max_body_rate_rad_s=0.4,
    )

    command = attitude_body_rate_command(
        (0.3, -0.1, 0.05),
        (0.0, 0.0, 0.0),
        cfg,
    )

    assert command == pytest.approx((-0.3, 0.1, -0.05))


def test_body_rate_loop_clears_deadband_with_minimum_command():
    cfg = AttitudeLoopConfig(
        kp_pitch=1.0,
        body_rate_gain_pitch=-2.0,
        body_rate_min_command_rad_s=0.02,
        attitude_error_deadband_rad=0.004,
    )

    active = attitude_body_rate_command((0.0, -0.01, 0.0), (0.0, 0.0, 0.0), cfg)
    quiet = attitude_body_rate_command((0.0, -0.003, 0.0), (0.0, 0.0, 0.0), cfg)

    assert active[1] == pytest.approx(0.02)
    assert quiet[1] == pytest.approx(0.0)


def test_ahrs_accel_blend_pulls_to_gravity():
    est = calibrated_estimator(velocity_leak_per_s=0.0, ahrs_accel_gain=0.2)
    est.state.pitch = 0.3  # wrong prior; drone actually level and at rest
    f_imu = REST_ACCEL
    for i in range(301):
        est.update_imu(i * 0.01, f_imu, (0.0, 0.0, 0.0))
    assert abs(est.state.pitch) < 0.02


def test_zero_velocity_update_clears_velocity():
    est = calibrated_estimator()
    est.state.velocity_ned_m_s = (1.0, -2.0, 0.5)
    est.zero_velocity_update()
    assert est.state.velocity_ned_m_s == (0.0, 0.0, 0.0)


def test_summary_shape():
    est = calibrated_estimator()
    est.observe_gate(0, (5.0, 0.0, 0.0))
    summary = est.to_summary()
    assert summary["calibrated"] is True
    assert "0" in summary["landmarks"]
    assert summary["imu_samples"] == 0
