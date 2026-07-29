#!/usr/bin/env python3
"""Competition-facing drone policy observation/action contract.

The native v4 env and the SITL controller both use this contract when training
or running policies intended to transfer through the TS-002 observable boundary.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from collections.abc import Sequence


OBSERVATION_FIELDS = (
    "gate_forward_rate_norm",
    "gate_right_rate_norm",
    "gate_down_rate_norm",
    "gyro_roll_rate_norm",
    "gyro_pitch_rate_norm",
    "gyro_yaw_rate_norm",
    "attitude_quat_w",
    "attitude_quat_x",
    "attitude_quat_y",
    "attitude_quat_z",
    "gate_visible",
    "gate_forward_norm",
    "gate_right_norm",
    "gate_down_norm",
    "gate_yaw_error_norm",
    "gate_pitch_error_norm",
    "gate_apparent_size_norm",
    "gate_centering_quality",
    "elapsed_time_fraction",
    "last_cmd_forward_norm",
    "last_cmd_right_norm",
    "last_cmd_down_norm",
    "last_cmd_yaw_rate_norm",
)

ACTION_FIELDS = (
    "cmd_forward_norm",
    "cmd_right_norm",
    "cmd_down_norm",
    "cmd_yaw_rate_norm",
)

# interface_mode=2 (DRONE_RACE_INTERFACE_ATTITUDE_SETPOINT): indices match
# move_drone_sitl_plant() — policy_action[0]=pitch, [1]=roll, [2]=thrust, [3]=yaw.
ATTITUDE_ACTION_FIELDS = (
    "cmd_pitch_norm",
    "cmd_roll_norm",
    "cmd_thrust_norm",
    "cmd_yaw_norm",
)

OBSERVATION_SIZE = len(OBSERVATION_FIELDS)
ACTION_SIZE = len(ACTION_FIELDS)
ATTITUDE_ACTION_SIZE = len(ATTITUDE_ACTION_FIELDS)

# Match ocean/drone_race/drone_race.c TS-002 observation encoding.
TS002_CAMERA_HALF_FOV_RAD = math.pi / 4.0
TS002_CAMERA_UPTILT_RAD = 0.0
TS002_GATE_INNER_WIDTH_M = 1.5
NATIVE_MAX_OMEGA_RAD_S = 20.0


def rotate_vector_by_quat(
    vector: Sequence[float], quat_wxyz: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate a three-vector by a unit quaternion."""
    if len(vector) != 3 or len(quat_wxyz) != 4:
        raise ValueError("expected a three-vector and a (w, x, y, z) quaternion")
    vx, vy, vz = (float(value) for value in vector)
    qw, qx, qy, qz = (float(value) for value in quat_wxyz)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


@dataclasses.dataclass(frozen=True)
class GateMotionFilterConfig:
    """Frame-rate-independent filtering for camera-derived gate motion.

    The v3385 camera arrives at about 15 Hz and detector range is quantized by
    pixel width.  Differentiating consecutive poses therefore creates large,
    native-only transfer errors.  Cascaded continuous-time low-pass filters
    retain the observable derivative while suppressing that quantization.
    """

    position_tau_s: float = 0.30
    rate_tau_s: float = 0.60
    max_sample_gap_s: float = 0.50
    max_innovation_base_m: float = 3.0
    max_innovation_speed_m_s: float = 30.0
    max_initial_range_m: float = 35.0
    predict_without_measurement: bool = False
    control_accel_gain_m_s2_per_tan_roll: float = 0.0
    reacquire_consecutive_samples: int = 0
    reacquire_max_candidate_jump_m: float = 3.0
    reacquire_closer_margin_m: float = 0.5


class ObservableGateMotionFilter:
    """Estimate gate-vector rates from camera poses and observable attitude."""

    def __init__(self, config: GateMotionFilterConfig | None = None) -> None:
        self.config = config or GateMotionFilterConfig()
        self.accepted_samples = 0
        self.rejected_samples = 0
        self.reacquired_samples = 0
        self.transition_preseeds = 0
        self.reset()

    def reset(self) -> None:
        self._identity = None
        self._pose_key = None
        self._raw_world: tuple[float, float, float] | None = None
        self._filtered_world: tuple[float, float, float] | None = None
        self._rate_world = (0.0, 0.0, 0.0)
        self._time_s: float | None = None
        self._last_accepted_time_s: float | None = None
        self._reacquire_candidate_world: tuple[float, float, float] | None = None
        self._reacquire_candidate_count = 0
        self._last_rejected_world: tuple[float, float, float] | None = None
        self._last_rejected_time_s: float | None = None
        self._last_rejected_identity = None

    def _clear_reacquire_candidate(self) -> None:
        self._reacquire_candidate_world = None
        self._reacquire_candidate_count = 0

    def preseed_identity_from_last_rejection(
        self,
        identity,
        *,
        time_s: float,
        max_age_s: float = 0.25,
        min_range_margin_m: float = 2.0,
    ) -> bool:
        """Carry a recent, farther rejected pose into an official gate change.

        Near the aperture, the detector can already see the next ordered gate.
        That farther pose is correctly rejected while the current official gate
        remains active, but it is often the best legal camera initialization for
        the next identity.  This method is deliberately explicit and disabled
        unless the runner requests it for a particular official transition.
        """
        now_s = _finite_float(time_s, field_name="gate_motion_time_s")
        max_age = _finite_float(max_age_s, field_name="gate_motion_max_age_s")
        min_margin = _finite_float(
            min_range_margin_m, field_name="gate_motion_min_range_margin_m")
        if max_age < 0.0 or min_margin < 0.0:
            raise ValueError("gate transition preseed bounds must be nonnegative")
        candidate = self._last_rejected_world
        rejected_time = self._last_rejected_time_s
        if (
            candidate is None
            or rejected_time is None
            or now_s - rejected_time < 0.0
            or now_s - rejected_time > max_age
            or self._last_rejected_identity == identity
            or self._raw_world is None
        ):
            return False
        candidate_range = math.sqrt(sum(value * value for value in candidate))
        tracked_range = math.sqrt(sum(value * value for value in self._raw_world))
        if (
            candidate_range > self.config.max_initial_range_m
            or candidate_range < tracked_range + min_margin
        ):
            return False

        carried_rate = self._rate_world
        self.reset()
        self._identity = identity
        self._raw_world = candidate
        self._filtered_world = candidate
        # Both ordered gates are stationary, so their relative world rates are
        # the same negated vehicle velocity at the identity transition.  Keep
        # that already-observed rate instead of inventing a zero-velocity step.
        self._rate_world = carried_rate
        self._time_s = now_s
        self._last_accepted_time_s = now_s
        self.accepted_samples += 1
        self.transition_preseeds += 1
        return True

    def _body_rate(self, quat_wxyz: Sequence[float]) -> tuple[float, float, float]:
        qw, qx, qy, qz = (float(value) for value in quat_wxyz)
        body_rate_up = rotate_vector_by_quat(
            self._rate_world, (qw, -qx, -qy, -qz)
        )
        return (body_rate_up[0], body_rate_up[1], -body_rate_up[2])

    def filtered_body_vector_ned(
        self, quat_wxyz: Sequence[float]
    ) -> tuple[float, float, float] | None:
        """Return the filtered gate pose in the current observable body frame."""
        if self._filtered_world is None:
            return None
        qw, qx, qy, qz = (float(value) for value in quat_wxyz)
        body_up = rotate_vector_by_quat(
            self._filtered_world, (qw, -qx, -qy, -qz)
        )
        return (body_up[0], body_up[1], -body_up[2])

    def tracked_body_vector_ned(
        self, quat_wxyz: Sequence[float]
    ) -> tuple[float, float, float] | None:
        """Return the last associated raw pose, holding across rejected aliases."""
        if self._raw_world is None:
            return None
        qw, qx, qy, qz = (float(value) for value in quat_wxyz)
        body_up = rotate_vector_by_quat(
            self._raw_world, (qw, -qx, -qy, -qz)
        )
        return (body_up[0], body_up[1], -body_up[2])

    def hold_without_measurement(
        self,
        quat_wxyz: Sequence[float],
        *,
        identity=None,
        time_s: float | None = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float] | None]:
        """Hold the last associated gate while its official identity is unchanged.

        Close gates routinely leave the detector FOV for a few frames. Clearing
        association on such a frame lets a distant red structure become the
        active gate. The official gate index is a stronger identity signal: only
        an index change is allowed to clear the tracked pose.
        """
        if self._raw_world is not None and identity != self._identity:
            self.reset()
        self._identity = identity
        if (
            self.config.predict_without_measurement
            and self._raw_world is not None
            and self._filtered_world is not None
            and self._time_s is not None
            and time_s is not None
        ):
            now_s = _finite_float(time_s, field_name="gate_motion_time_s")
            dt = now_s - self._time_s
            if dt > 0.0:
                previous_rate = self._rate_world
                gain = self.config.control_accel_gain_m_s2_per_tan_roll
                if gain != 0.0:
                    qw, qx, qy, qz = (float(value) for value in quat_wxyz)
                    roll = math.atan2(
                        2.0 * (qw * qx + qy * qz),
                        1.0 - 2.0 * (qx * qx + qy * qy),
                    )
                    lateral_accel = gain * math.tan(
                        clamp(roll, -1.2, 1.2)
                    )
                    self._rate_world = (
                        previous_rate[0],
                        previous_rate[1] + lateral_accel * dt,
                        previous_rate[2],
                    )
                delta = tuple(
                    0.5 * (previous_rate[index] + self._rate_world[index]) * dt
                    for index in range(3)
                )
                self._raw_world = tuple(
                    self._raw_world[index] + delta[index] for index in range(3)
                )
                self._filtered_world = tuple(
                    self._filtered_world[index] + delta[index]
                    for index in range(3)
                )
                self._time_s = now_s
        return self._body_rate(quat_wxyz), self.tracked_body_vector_ned(quat_wxyz)

    def update(
        self,
        body_vector_ned_m: Sequence[float],
        quat_wxyz: Sequence[float],
        time_s: float,
        *,
        identity=None,
    ) -> tuple[float, float, float]:
        """Return `(forward, right, down)` gate-vector rates in m/s.

        Repeated calls for the same camera pose do not create artificial zero
        velocity samples. `identity` should be the official active gate index
        when available so a gate transition resets the filter deterministically.
        """
        if len(body_vector_ned_m) != 3 or len(quat_wxyz) != 4:
            raise ValueError("invalid gate-vector or quaternion size")
        now_s = _finite_float(time_s, field_name="gate_motion_time_s")
        body = tuple(
            _finite_float(value, field_name="gate_motion_body_vector")
            for value in body_vector_ned_m
        )
        quat = tuple(
            _finite_float(value, field_name="gate_motion_quaternion")
            for value in quat_wxyz
        )
        pose_key = body
        if self._raw_world is not None and identity != self._identity:
            self.reset()
        self._identity = identity
        if pose_key == self._pose_key:
            if self.config.predict_without_measurement:
                rate, _ = self.hold_without_measurement(
                    quat, identity=identity, time_s=now_s
                )
                return rate
            return self._body_rate(quat)

        range_m = math.sqrt(sum(value * value for value in body))
        if self._raw_world is None and range_m > self.config.max_initial_range_m:
            # A target gate should be the nearest usable gate. In official
            # traces, 40-75 m first detections are other red structures/gates;
            # accepting one before association poisons every subsequent pose.
            self._pose_key = pose_key
            self.rejected_samples += 1
            return (0.0, 0.0, 0.0)

        # The native dynamics use Z-up while the official body-vector contract
        # is NED. Convert to the shared Z-up vector before attitude compensation.
        world = rotate_vector_by_quat((body[0], body[1], -body[2]), quat)
        self._pose_key = pose_key
        if self._raw_world is None or self._time_s is None:
            self._raw_world = world
            self._filtered_world = world
            self._rate_world = (0.0, 0.0, 0.0)
            self._time_s = now_s
            self._last_accepted_time_s = now_s
            self.accepted_samples += 1
            return (0.0, 0.0, 0.0)

        dt = now_s - self._time_s
        cfg = self.config
        if dt <= 0.0:
            return self._body_rate(quat)

        # A long interval does not erase gate identity. In particular, do not
        # accept a 35-75 m structure after briefly losing a 3-5 m gate merely
        # because the detector gap exceeded half a second. Cap the association
        # horizon and require an explicit official gate-index change to reset.
        association_elapsed_s = now_s - (
            self._last_accepted_time_s
            if self._last_accepted_time_s is not None
            else self._time_s
        )
        association_dt = min(association_elapsed_s, cfg.max_sample_gap_s)

        innovation = math.sqrt(
            sum((world[index] - self._raw_world[index]) ** 2 for index in range(3))
        )
        innovation_limit = max(
            cfg.max_innovation_base_m,
            cfg.max_innovation_speed_m_s * association_dt,
        )
        if innovation > innovation_limit:
            # Remember the pose key so the same bad video frame is rejected
            # once, but retain the previous good sample and its time baseline.
            self.rejected_samples += 1
            self._last_rejected_world = world
            self._last_rejected_time_s = now_s
            self._last_rejected_identity = identity
            reacquire_enabled = cfg.reacquire_consecutive_samples > 0
            tracked_range_m = math.sqrt(
                sum(value * value for value in self._raw_world)
            )
            candidate_is_nearer = (
                range_m <= cfg.max_initial_range_m
                and range_m
                <= tracked_range_m - cfg.reacquire_closer_margin_m
            )
            if reacquire_enabled and candidate_is_nearer:
                candidate_jump_m = (
                    0.0
                    if self._reacquire_candidate_world is None
                    else math.sqrt(
                        sum(
                            (
                                world[index]
                                - self._reacquire_candidate_world[index]
                            )
                            ** 2
                            for index in range(3)
                        )
                    )
                )
                if (
                    self._reacquire_candidate_world is None
                    or candidate_jump_m
                    > cfg.reacquire_max_candidate_jump_m
                ):
                    self._reacquire_candidate_count = 1
                else:
                    self._reacquire_candidate_count += 1
                self._reacquire_candidate_world = world
                if (
                    self._reacquire_candidate_count
                    >= cfg.reacquire_consecutive_samples
                ):
                    # The official identity is unchanged, but consecutive
                    # nearer detections agree with each other and contradict
                    # the propagated track. Re-anchor position while retaining
                    # the bounded velocity estimate; this recovers the active
                    # gate without ever accepting a farther red structure.
                    self._raw_world = world
                    self._filtered_world = world
                    self._time_s = now_s
                    self._last_accepted_time_s = now_s
                    self.accepted_samples += 1
                    self.reacquired_samples += 1
                    self._clear_reacquire_candidate()
                    return self._body_rate(quat)
            else:
                self._clear_reacquire_candidate()
            if cfg.predict_without_measurement:
                rate, _ = self.hold_without_measurement(
                    quat, identity=identity, time_s=now_s
                )
                return rate
            return self._body_rate(quat)

        assert self._filtered_world is not None
        # Association and state propagation have different clocks. The last
        # accepted camera time provides enough horizon to admit a physically
        # reachable measurement after 60 Hz prediction ticks. Once admitted,
        # preserve the historical predictor/filter dynamics by applying the
        # correction over the time since the last propagated state. Feeding
        # association_dt into these filters understates live closing speed and
        # changes the already-proven H12 Gates-1--3 control law.
        filter_dt = dt
        position_alpha = 1.0 - math.exp(
            -filter_dt / max(cfg.position_tau_s, 1e-6)
        )
        filtered_world = tuple(
            self._filtered_world[index]
            + position_alpha * (world[index] - self._filtered_world[index])
            for index in range(3)
        )
        measured_rate = tuple(
            (filtered_world[index] - self._filtered_world[index]) / filter_dt
            for index in range(3)
        )
        rate_alpha = 1.0 - math.exp(
            -filter_dt / max(cfg.rate_tau_s, 1e-6)
        )
        self._rate_world = tuple(
            self._rate_world[index]
            + rate_alpha * (measured_rate[index] - self._rate_world[index])
            for index in range(3)
        )
        self._raw_world = world
        self._filtered_world = filtered_world
        self._time_s = now_s
        self._last_accepted_time_s = now_s
        self.accepted_samples += 1
        self._clear_reacquire_candidate()
        return self._body_rate(quat)

    def snapshot(self) -> dict:
        return {
            "config": dataclasses.asdict(self.config),
            "accepted_samples": self.accepted_samples,
            "rejected_samples": self.rejected_samples,
            "reacquired_samples": self.reacquired_samples,
            "transition_preseeds": self.transition_preseeds,
            "reacquire_candidate_count": self._reacquire_candidate_count,
            "initialized": self._raw_world is not None,
            "identity": self._identity,
            "tracked_world": (
                None
                if self._raw_world is None
                else tuple(round(value, 6) for value in self._raw_world)
            ),
            "filtered_world": (
                None
                if self._filtered_world is None
                else tuple(round(value, 6) for value in self._filtered_world)
            ),
            "rate_world": tuple(round(value, 6) for value in self._rate_world),
        }


def _tanh_norm(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return clamp(math.tanh(value / scale), -1.0, 1.0)


def build_ts002_observation(
    *,
    guidance_visible: bool = False,
    guidance_center_x_norm: float = 0.0,
    guidance_heading_error_rad: float = 0.0,
    gate_forward_rate_m_s: float = 0.0,
    gate_right_rate_m_s: float = 0.0,
    gate_down_rate_m_s: float = 0.0,
    gyro_roll_rad_s: float = 0.0,
    gyro_pitch_rad_s: float = 0.0,
    gyro_yaw_rad_s: float = 0.0,
    quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    gate_visible: bool = False,
    gate_forward_m: float = 0.0,
    gate_right_m: float = 0.0,
    gate_down_m: float = 0.0,
    gate_yaw_error_rad: float = 0.0,
    gate_range_m: float | None = None,
    elapsed_fraction: float = 0.0,
    last_cmd: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    max_omega_rad_s: float = NATIVE_MAX_OMEGA_RAD_S,
) -> tuple[float, ...]:
    """Build the 23-float observation vector used by native drone_race mode 1/2."""
    if len(quat_wxyz) != 4:
        raise ValueError("quat_wxyz must be (w, x, y, z)")
    if len(last_cmd) != 4:
        raise ValueError("last_cmd must have four normalized action values")

    omega_scale = max(max_omega_rad_s, 1e-6)
    observation = [
        _tanh_norm(gate_forward_rate_m_s, 5.0),
        _tanh_norm(gate_right_rate_m_s, 3.0),
        _tanh_norm(gate_down_rate_m_s, 3.0),
        clamp(gyro_roll_rad_s / omega_scale, -1.0, 1.0),
        clamp(gyro_pitch_rad_s / omega_scale, -1.0, 1.0),
        clamp(gyro_yaw_rad_s / omega_scale, -1.0, 1.0),
        clamp(float(quat_wxyz[0]), -1.0, 1.0),
        clamp(float(quat_wxyz[1]), -1.0, 1.0),
        clamp(float(quat_wxyz[2]), -1.0, 1.0),
        clamp(float(quat_wxyz[3]), -1.0, 1.0),
    ]

    if gate_visible:
        forward = gate_forward_m
        right = gate_right_m
        down = gate_down_m
        range_m = gate_range_m
        if range_m is None:
            range_m = math.sqrt(forward * forward + right * right + down * down)
        range_m = max(range_m, 1e-3)
        elevation = math.atan2(-down, max(forward, 1e-4))
        camera_pitch_error = elevation - TS002_CAMERA_UPTILT_RAD
        yaw_score = 1.0 - abs(gate_yaw_error_rad) / TS002_CAMERA_HALF_FOV_RAD
        pitch_score = 1.0 - abs(camera_pitch_error) / TS002_CAMERA_HALF_FOV_RAD
        confidence = clamp(0.5 * (yaw_score + pitch_score), 0.0, 1.0)

        observation.extend(
            [
                1.0,
                _tanh_norm(forward, 10.0),
                _tanh_norm(right, 5.0),
                _tanh_norm(down, 5.0),
                clamp(gate_yaw_error_rad / TS002_CAMERA_HALF_FOV_RAD, -1.0, 1.0),
                clamp(camera_pitch_error / TS002_CAMERA_HALF_FOV_RAD, -1.0, 1.0),
                clamp(TS002_GATE_INNER_WIDTH_M / range_m, 0.0, 1.0),
                confidence,
            ]
        )
    else:
        observation.extend([0.0] * 8)

    observation.append(clamp(float(elapsed_fraction), 0.0, 1.0))
    observation.extend(clamp(float(last_cmd[i]), -1.0, 1.0) for i in range(4))
    return validate_observation(observation)


@dataclasses.dataclass(frozen=True)
class PolicyActionScales:
    max_forward_m_s: float = 2.0
    max_right_m_s: float = 1.0
    max_down_m_s: float = 0.8
    max_yaw_rate_rad_s: float = 1.0


@dataclasses.dataclass(frozen=True)
class PolicyAction:
    normalized: tuple[float, float, float, float]
    forward_m_s: float
    right_m_s: float
    down_m_s: float
    yaw_rate_rad_s: float


@dataclasses.dataclass(frozen=True)
class MavlinkVelocityYawSetpoint:
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    yaw_rate_rad_s: float


@dataclasses.dataclass(frozen=True)
class AttitudeActionScales:
    max_pitch_rad: float = 0.5
    max_roll_rad: float = 0.5
    max_yaw_rad: float = math.pi
    hover_thrust: float = 0.27
    min_thrust: float = 0.18
    max_thrust: float = 0.42


@dataclasses.dataclass(frozen=True)
class AttitudePolicyAction:
    normalized: tuple[float, float, float, float]
    pitch_rad: float
    roll_rad: float
    thrust: float
    yaw_rad: float


@dataclasses.dataclass(frozen=True)
class AttitudeSetpoint:
    pitch_rad: float
    roll_rad: float
    thrust: float
    yaw_rad: float


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _finite_float(value, *, field_name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    return out


def normalize_action(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != ACTION_SIZE:
        raise ValueError(f"expected {ACTION_SIZE} action values")
    return tuple(clamp(_finite_float(value, field_name=ACTION_FIELDS[i]), -1.0, 1.0)
        for i, value in enumerate(values))


def normalize_attitude_action(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != ATTITUDE_ACTION_SIZE:
        raise ValueError(f"expected {ATTITUDE_ACTION_SIZE} attitude action values")
    out = []
    for i, value in enumerate(values):
        if i == 2:
            out.append(clamp(_finite_float(value, field_name=ATTITUDE_ACTION_FIELDS[i]), -1.0, 1.0))
        else:
            out.append(clamp(_finite_float(value, field_name=ATTITUDE_ACTION_FIELDS[i]), -1.0, 1.0))
    return tuple(out)


def decode_attitude_policy_action(
    values: Sequence[float],
    *,
    scales: AttitudeActionScales | None = None,
) -> tuple[AttitudePolicyAction, AttitudeSetpoint]:
    scales = scales or AttitudeActionScales()
    action = normalize_attitude_action(values)
    pitch = action[0] * scales.max_pitch_rad
    roll = action[1] * scales.max_roll_rad
    thrust_span = (
        scales.max_thrust - scales.hover_thrust
        if action[2] >= 0.0
        else scales.hover_thrust - scales.min_thrust
    )
    thrust = scales.hover_thrust + action[2] * thrust_span
    yaw = action[3] * scales.max_yaw_rad
    return (
        AttitudePolicyAction(
            normalized=action,
            pitch_rad=pitch,
            roll_rad=roll,
            thrust=thrust,
            yaw_rad=yaw,
        ),
        AttitudeSetpoint(
            pitch_rad=pitch,
            roll_rad=roll,
            thrust=thrust,
            yaw_rad=yaw,
        ),
    )


def attitude_last_cmd_normalized(action: AttitudePolicyAction) -> tuple[float, float, float, float]:
    """Map attitude action to observation last-cmd slots 19-22 (mode 2 contract)."""
    return action.normalized


def normalize_attitude_thrust(
    thrust: float,
    *,
    scales: AttitudeActionScales | None = None,
) -> float:
    """Encode physical thrust using the hover-centered mode-2 action contract."""
    scales = scales or AttitudeActionScales()
    thrust = _finite_float(thrust, field_name="thrust")
    span = (
        scales.max_thrust - scales.hover_thrust
        if thrust >= scales.hover_thrust
        else scales.hover_thrust - scales.min_thrust
    )
    return clamp((thrust - scales.hover_thrust) / max(span, 1e-9), -1.0, 1.0)


def decode_policy_action(
    values: Sequence[float],
    *,
    yaw_rad: float,
    scales: PolicyActionScales | None = None,
) -> tuple[PolicyAction, MavlinkVelocityYawSetpoint]:
    scales = scales or PolicyActionScales()
    action = normalize_action(values)
    forward = action[0] * scales.max_forward_m_s
    right = action[1] * scales.max_right_m_s
    down = action[2] * scales.max_down_m_s
    yaw_rate = action[3] * scales.max_yaw_rate_rad_s

    yaw = _finite_float(yaw_rad, field_name="yaw_rad")
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx = c * forward - s * right
    vy = s * forward + c * right

    return (
        PolicyAction(
            normalized=action,
            forward_m_s=forward,
            right_m_s=right,
            down_m_s=down,
            yaw_rate_rad_s=yaw_rate,
        ),
        MavlinkVelocityYawSetpoint(
            vx_m_s=vx,
            vy_m_s=vy,
            vz_m_s=down,
            yaw_rate_rad_s=yaw_rate,
        ),
    )


def validate_observation(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != OBSERVATION_SIZE:
        raise ValueError(f"expected {OBSERVATION_SIZE} observation values")
    out = tuple(_finite_float(value, field_name=OBSERVATION_FIELDS[i])
        for i, value in enumerate(values))
    bad = [
        OBSERVATION_FIELDS[i]
        for i, value in enumerate(out)
        if value < -1.000001 or value > 1.000001
    ]
    if bad:
        raise ValueError(f"observation fields outside [-1, 1]: {', '.join(bad)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a normalized drone policy action")
    parser.add_argument("--action-json", required=True, help="JSON list of four normalized action values")
    parser.add_argument("--yaw-rad", type=float, default=0.0)
    parser.add_argument("--json-path", default="")
    args = parser.parse_args()

    action, setpoint = decode_policy_action(json.loads(args.action_json), yaw_rad=args.yaw_rad)
    report = {
        "action": dataclasses.asdict(action),
        "setpoint": dataclasses.asdict(setpoint),
        "observation_fields": OBSERVATION_FIELDS,
        "action_fields": ACTION_FIELDS,
    }
    if args.json_path:
        directory = os.path.dirname(args.json_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
