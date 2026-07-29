from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "drone_policy_contract.py"
spec = importlib.util.spec_from_file_location("drone_policy_contract", MODULE_PATH)
contract = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = contract
spec.loader.exec_module(contract)


def test_contract_sizes_and_field_names():
    assert contract.OBSERVATION_SIZE == 23
    assert contract.ACTION_SIZE == 4
    assert contract.OBSERVATION_FIELDS[10:18] == (
        "gate_visible",
        "gate_forward_norm",
        "gate_right_norm",
        "gate_down_norm",
        "gate_yaw_error_norm",
        "gate_pitch_error_norm",
        "gate_apparent_size_norm",
        "gate_centering_quality",
    )
    assert contract.ACTION_FIELDS == (
        "cmd_forward_norm",
        "cmd_right_norm",
        "cmd_down_norm",
        "cmd_yaw_rate_norm",
    )
    assert contract.OBSERVATION_FIELDS[:3] == (
        "gate_forward_rate_norm",
        "gate_right_rate_norm",
        "gate_down_rate_norm",
    )


def test_decode_policy_action_scales_and_rotates_body_to_local_ned():
    action, setpoint = contract.decode_policy_action(
        [2.0, -0.5, 0.25, -2.0],
        yaw_rad=math.pi / 2.0,
    )

    assert action.normalized == (1.0, -0.5, 0.25, -1.0)
    assert action.forward_m_s == pytest.approx(2.0)
    assert action.right_m_s == pytest.approx(-0.5)
    assert action.down_m_s == pytest.approx(0.2)
    assert action.yaw_rate_rad_s == pytest.approx(-1.0)
    assert setpoint.vx_m_s == pytest.approx(0.5)
    assert setpoint.vy_m_s == pytest.approx(2.0)
    assert setpoint.vz_m_s == pytest.approx(0.2)
    assert setpoint.yaw_rate_rad_s == pytest.approx(-1.0)


def test_validate_observation_rejects_wrong_shape_and_bounds():
    obs = [0.0] * contract.OBSERVATION_SIZE
    assert contract.validate_observation(obs) == tuple(obs)

    with pytest.raises(ValueError, match="expected 23"):
        contract.validate_observation(obs[:-1])

    obs[3] = 1.1
    with pytest.raises(ValueError, match="gyro_roll_rate_norm"):
        contract.validate_observation(obs)


def test_decode_attitude_policy_action_matches_mode2_layout():
    action, setpoint = contract.decode_attitude_policy_action(
        [0.5, -0.25, 0.0, 0.5],
    )
    assert action.normalized == (0.5, -0.25, 0.0, 0.5)
    assert action.pitch_rad == pytest.approx(0.25)
    assert action.roll_rad == pytest.approx(-0.125)
    assert action.thrust == pytest.approx(0.27)
    assert action.yaw_rad == pytest.approx(math.pi * 0.5)
    assert setpoint.pitch_rad == action.pitch_rad
    assert contract.attitude_last_cmd_normalized(action) == action.normalized
    assert contract.normalize_attitude_thrust(0.27) == pytest.approx(0.0)
    assert contract.normalize_attitude_thrust(0.42) == pytest.approx(1.0)
    assert contract.normalize_attitude_thrust(0.18) == pytest.approx(-1.0)
    assert contract.decode_attitude_policy_action([0, 0, 1, 0])[1].thrust == pytest.approx(0.42)
    assert contract.decode_attitude_policy_action([0, 0, -1, 0])[1].thrust == pytest.approx(0.18)
    assert contract.ATTITUDE_ACTION_FIELDS == (
        "cmd_pitch_norm",
        "cmd_roll_norm",
        "cmd_thrust_norm",
        "cmd_yaw_norm",
    )


def test_build_ts002_observation_matches_native_visible_gate_encoding():
    obs = contract.build_ts002_observation(
        guidance_visible=True,
        guidance_center_x_norm=-0.5,
        guidance_heading_error_rad=0.1,
        gate_forward_rate_m_s=-2.5,
        gate_right_rate_m_s=1.5,
        gate_down_rate_m_s=-0.75,
        gyro_roll_rad_s=2.0,
        gyro_pitch_rad_s=-1.0,
        gyro_yaw_rad_s=0.5,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        gate_visible=True,
        gate_forward_m=2.0,
        gate_right_m=0.0,
        gate_down_m=0.0,
        gate_yaw_error_rad=0.1,
        gate_range_m=7.5,
        elapsed_fraction=0.25,
        last_cmd=(0.1, -0.2, 0.3, -0.4),
    )
    assert len(obs) == contract.OBSERVATION_SIZE
    assert obs[0] == pytest.approx(math.tanh(-0.5))
    assert obs[1] == pytest.approx(math.tanh(0.5))
    assert obs[2] == pytest.approx(math.tanh(-0.25))
    assert obs[10] == 1.0
    assert obs[16] == pytest.approx(contract.TS002_GATE_INNER_WIDTH_M / 7.5, rel=1e-4)
    expected_centering = 1.0 - 0.5 * (
        abs(obs[14]) + abs(obs[15])
    )
    assert obs[17] == pytest.approx(expected_centering)
    assert obs[18] == pytest.approx(0.25)
    assert obs[19] == pytest.approx(0.1)


def test_build_ts002_observation_hides_gate_features_when_not_visible():
    obs = contract.build_ts002_observation(gate_visible=False)
    assert obs[10] == 0.0
    assert all(value == 0.0 for value in obs[11:18])


def test_gate_motion_filter_converges_and_is_frame_rate_independent():
    def run(dt):
        motion = contract.ObservableGateMotionFilter()
        rate = (0.0, 0.0, 0.0)
        steps = int(5.0 / dt)
        for index in range(steps + 1):
            time_s = index * dt
            rate = motion.update(
                (20.0 - 2.0 * time_s, 0.5 * time_s, -0.25 * time_s),
                (1.0, 0.0, 0.0, 0.0),
                time_s,
                identity=0,
            )
        return rate

    slow = run(1.0 / 15.0)
    fast = run(1.0 / 120.0)
    assert slow == pytest.approx((-2.0, 0.5, -0.25), abs=0.03)
    assert fast == pytest.approx(slow, abs=0.03)


def test_gate_motion_filter_ignores_duplicate_frames_and_range_outlier():
    motion = contract.ObservableGateMotionFilter()
    quat = (1.0, 0.0, 0.0, 0.0)
    motion.update((20.0, 0.0, 0.0), quat, 0.0, identity=0)
    first = motion.update((19.8, 0.0, 0.0), quat, 0.1, identity=0)
    duplicate = motion.update((19.8, 0.0, 0.0), quat, 0.15, identity=0)
    outlier = motion.update((40.0, 0.0, 0.0), quat, 0.2, identity=0)

    assert duplicate == pytest.approx(first)
    assert outlier == pytest.approx(first)
    assert motion.snapshot()["rejected_samples"] == 1
    filtered = motion.filtered_body_vector_ned(quat)
    assert filtered is not None
    assert filtered[0] < 20.0
    assert filtered[0] > 19.8
    tracked = motion.tracked_body_vector_ned(quat)
    assert tracked == pytest.approx((19.8, 0.0, 0.0))

    reset_rate = motion.update((5.0, 0.0, 0.0), quat, 0.3, identity=1)
    assert reset_rate == (0.0, 0.0, 0.0)


def test_gate_motion_filter_rejects_far_first_alias_before_association():
    motion = contract.ObservableGateMotionFilter()
    quat = (1.0, 0.0, 0.0, 0.0)
    assert motion.update((48.0, 2.0, 6.0), quat, 0.0, identity=0) == (0.0, 0.0, 0.0)
    assert motion.tracked_body_vector_ned(quat) is None
    assert motion.snapshot()["rejected_samples"] == 1

    motion.update((22.0, 0.2, -0.3), quat, 0.1, identity=0)
    assert motion.tracked_body_vector_ned(quat) == pytest.approx((22.0, 0.2, -0.3))


def test_gate_motion_filter_does_not_reassociate_far_alias_after_sample_gap():
    motion = contract.ObservableGateMotionFilter()
    quat = (1.0, 0.0, 0.0, 0.0)
    motion.update((3.74, -2.89, 0.66), quat, 8.984, identity=2)

    # Official gate-3 failure signature: the close gate leaves the FOV and a
    # distant red structure persists beyond max_sample_gap_s.
    motion.update((40.0, 5.25, 5.0), quat, 9.093, identity=2)
    motion.update((41.74, 6.26, 5.41), quat, 9.328, identity=2)
    motion.update((34.29, 6.64, 4.71), quat, 9.609, identity=2)

    assert motion.tracked_body_vector_ned(quat) == pytest.approx(
        (3.74, -2.89, 0.66)
    )
    assert motion.snapshot()["accepted_samples"] == 1
    assert motion.snapshot()["rejected_samples"] == 3


def test_gate_motion_filter_holds_missing_pose_until_identity_changes():
    motion = contract.ObservableGateMotionFilter()
    quat = (1.0, 0.0, 0.0, 0.0)
    motion.update((5.0, -1.0, 0.25), quat, 1.0, identity=2)

    rate, held = motion.hold_without_measurement(quat, identity=2)
    assert rate == pytest.approx((0.0, 0.0, 0.0))
    assert held == pytest.approx((5.0, -1.0, 0.25))

    _, cleared = motion.hold_without_measurement(quat, identity=3)
    assert cleared is None


def test_gate_motion_filter_can_predict_pose_during_missing_measurement():
    motion = contract.ObservableGateMotionFilter(
        contract.GateMotionFilterConfig(predict_without_measurement=True)
    )
    quat = (1.0, 0.0, 0.0, 0.0)
    motion.update((5.0, 0.0, 0.0), quat, 1.0, identity=2)
    motion.update((4.0, 0.0, 0.0), quat, 1.2, identity=2)

    rate, predicted = motion.hold_without_measurement(
        quat, identity=2, time_s=1.4
    )

    assert rate[0] < 0.0
    assert predicted[0] < 4.0
    assert motion.snapshot()["config"]["predict_without_measurement"] is True


def test_gate_motion_filter_can_integrate_observable_roll_acceleration():
    motion = contract.ObservableGateMotionFilter(
        contract.GateMotionFilterConfig(
            predict_without_measurement=True,
            control_accel_gain_m_s2_per_tan_roll=4.295,
        )
    )
    identity = (1.0, 0.0, 0.0, 0.0)
    motion.update((10.0, 2.0, 0.0), identity, 1.0, identity=3)
    roll = -0.4
    quat = (math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0)

    motion.hold_without_measurement(quat, identity=3, time_s=1.5)
    snapshot = motion.snapshot()

    assert snapshot["rate_world"][1] < -0.8
    assert snapshot["tracked_world"][1] < 2.0
    assert (
        snapshot["config"]["control_accel_gain_m_s2_per_tan_roll"]
        == pytest.approx(4.295)
    )


def test_gate_motion_filter_reacquires_two_consistent_nearer_samples():
    motion = contract.ObservableGateMotionFilter(
        contract.GateMotionFilterConfig(
            max_innovation_base_m=1.0,
            max_innovation_speed_m_s=1.0,
            reacquire_consecutive_samples=2,
            reacquire_max_candidate_jump_m=1.0,
            reacquire_closer_margin_m=0.5,
        )
    )
    quat = (1.0, 0.0, 0.0, 0.0)
    motion.update((20.0, 0.0, 0.0), quat, 1.0, identity=0)

    motion.update((12.0, 2.0, 0.0), quat, 1.1, identity=0)
    assert motion.tracked_body_vector_ned(quat) == pytest.approx((20.0, 0.0, 0.0))
    motion.update((11.5, 2.1, 0.0), quat, 1.2, identity=0)

    assert motion.tracked_body_vector_ned(quat) == pytest.approx((11.5, 2.1, 0.0))
    snapshot = motion.snapshot()
    assert snapshot["reacquired_samples"] == 1
    assert snapshot["reacquire_candidate_count"] == 0


def test_gate_motion_filter_never_reacquires_farther_aliases():
    motion = contract.ObservableGateMotionFilter(
        contract.GateMotionFilterConfig(
            max_innovation_base_m=1.0,
            max_innovation_speed_m_s=1.0,
            reacquire_consecutive_samples=2,
        )
    )
    quat = (1.0, 0.0, 0.0, 0.0)
    motion.update((10.0, 0.0, 0.0), quat, 1.0, identity=0)
    motion.update((40.0, 0.0, 0.0), quat, 1.1, identity=0)
    motion.update((39.5, 0.0, 0.0), quat, 1.2, identity=0)

    assert motion.tracked_body_vector_ned(quat) == pytest.approx((10.0, 0.0, 0.0))
    assert motion.snapshot()["reacquired_samples"] == 0


def test_gate_motion_filter_can_preseed_next_identity_from_recent_far_rejection():
    motion = contract.ObservableGateMotionFilter()
    quat = (1.0, 0.0, 0.0, 0.0)
    current_gate = (2.0, 0.1, 0.0)
    visible_next_gate = (14.7, 8.6, -1.2)
    motion.update(current_gate, quat, 1.0, identity=0)
    current_gate = (1.8, 0.1, 0.0)
    motion.update(current_gate, quat, 1.1, identity=0)
    carried_rate = motion.snapshot()["rate_world"]
    motion.update(visible_next_gate, quat, 1.2, identity=0)

    assert motion.tracked_body_vector_ned(quat) == pytest.approx(current_gate)
    assert motion.preseed_identity_from_last_rejection(
        1, time_s=1.25, max_age_s=0.25, min_range_margin_m=2.0)
    assert motion.tracked_body_vector_ned(quat) == pytest.approx(visible_next_gate)
    assert motion.snapshot()["identity"] == 1
    assert motion.snapshot()["rate_world"] == carried_rate
    assert motion.snapshot()["transition_preseeds"] == 1


def test_gate_motion_filter_rejects_stale_transition_preseed():
    motion = contract.ObservableGateMotionFilter()
    quat = (1.0, 0.0, 0.0, 0.0)
    motion.update((2.0, 0.0, 0.0), quat, 1.0, identity=0)
    motion.update((14.0, 8.0, 0.0), quat, 1.1, identity=0)

    assert not motion.preseed_identity_from_last_rejection(
        1, time_s=1.5, max_age_s=0.25)
    assert motion.tracked_body_vector_ned(quat) == pytest.approx((2.0, 0.0, 0.0))
    assert motion.snapshot()["transition_preseeds"] == 0


def test_prediction_does_not_shrink_last_good_association_horizon():
    motion = contract.ObservableGateMotionFilter(
        contract.GateMotionFilterConfig(
            predict_without_measurement=True,
            max_innovation_base_m=1.0,
            max_innovation_speed_m_s=10.0,
            max_sample_gap_s=0.5,
        )
    )
    quat = (1.0, 0.0, 0.0, 0.0)
    motion.update((20.0, 0.0, 0.0), quat, 1.0, identity=0)
    motion.hold_without_measurement(quat, identity=0, time_s=1.1)
    motion.hold_without_measurement(quat, identity=0, time_s=1.2)
    motion.hold_without_measurement(quat, identity=0, time_s=1.3)

    motion.update((16.5, 0.0, 0.0), quat, 1.4, identity=0)

    assert motion.tracked_body_vector_ned(quat) == pytest.approx((16.5, 0.0, 0.0))
    assert motion.snapshot()["accepted_samples"] == 2


def test_association_horizon_does_not_replace_filter_propagation_timestep():
    motion = contract.ObservableGateMotionFilter(
        contract.GateMotionFilterConfig(
            predict_without_measurement=True,
            max_innovation_base_m=1.0,
            max_innovation_speed_m_s=10.0,
            max_sample_gap_s=0.5,
        )
    )
    quat = (1.0, 0.0, 0.0, 0.0)
    motion.update((20.0, 0.0, 0.0), quat, 1.0, identity=0)
    motion.hold_without_measurement(quat, identity=0, time_s=1.1)
    motion.hold_without_measurement(quat, identity=0, time_s=1.2)
    motion.hold_without_measurement(quat, identity=0, time_s=1.3)
    motion.update((16.5, 0.0, 0.0), quat, 1.4, identity=0)

    # The 0.4 s last-good horizon admits the 3.5 m innovation, while the
    # position/rate correction retains the 0.1 s propagation timestep.
    filtered = motion.filtered_body_vector_ned(quat)
    assert filtered is not None
    assert filtered[0] == pytest.approx(19.007860, abs=1e-6)
    assert motion.snapshot()["rate_world"][0] == pytest.approx(-1.523117, abs=1e-6)


def test_repeated_camera_pose_is_prediction_not_a_zero_velocity_measurement():
    motion = contract.ObservableGateMotionFilter(
        contract.GateMotionFilterConfig(
            predict_without_measurement=True,
            control_accel_gain_m_s2_per_tan_roll=4.295,
        )
    )
    identity = (1.0, 0.0, 0.0, 0.0)
    pose = (10.0, 2.0, 0.0)
    motion.update(pose, identity, 1.0, identity=3)
    roll = -0.4
    quat = (math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0)

    motion.update(pose, quat, 1.5, identity=3)

    assert motion.snapshot()["rate_world"][1] < -0.8
    assert motion.snapshot()["tracked_world"][1] < 2.0


def test_gate_transition_cache_clear_prevents_prior_gate_poisoning():
    quat = (1.0, 0.0, 0.0, 0.0)
    stale_previous_gate = (3.772607, -2.360320, 0.221550)
    visible_new_gate = (24.615385, -3.192308, 5.538462)

    poisoned = contract.ObservableGateMotionFilter()
    poisoned.update(stale_previous_gate, quat, 7.50, identity=2)
    poisoned.update(visible_new_gate, quat, 7.61, identity=2)
    assert poisoned.tracked_body_vector_ned(quat) == pytest.approx(
        stale_previous_gate
    )

    corrected = contract.ObservableGateMotionFilter()
    corrected.update(visible_new_gate, quat, 7.61, identity=2)
    assert corrected.tracked_body_vector_ned(quat) == pytest.approx(
        visible_new_gate
    )
    assert corrected.snapshot()["rejected_samples"] == 0
