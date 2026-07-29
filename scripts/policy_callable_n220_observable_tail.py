#!/usr/bin/env python3
"""N220: proven H12 prefix with an observable course controller on Gates 4--6.

The recurrent N219 tail still supplies pitch and yaw.  Roll and thrust are
replaced after official Gate 3 by a stateful controller whose only runtime
inputs are the deployed 32-value observation and the fixed course prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, atanh, cos, exp, log, sin, sqrt
from typing import Sequence

import numpy as np

import policy_callable_n219_unified_tail as n219
import policy_callable_six_gate_composite as six_gate


DT = 1.0 / 60.0
NUM_GATES = 6
GATES = np.asarray(
    (
        (22.35, 0.19, 0.30),
        (43.77, 2.04, -3.45),
        (67.53733, -1.24074, -10.23080),
        (95.63180, 5.10711, -18.24322),
        (122.0, 5.10711, -18.24322),
        (148.0, 5.10711, -18.24322),
    ),
    dtype=np.float64,
)
COURSE_ORIGIN = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
ROLL_BIAS = np.asarray((0.0, 0.0, 0.0, -0.40, 0.10, -0.12), dtype=np.float64)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _unscale_tanh(value: float, scale: float) -> float:
    return atanh(_clamp(float(value), -0.9999, 0.9999)) / scale


def _quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = float(q[0])
    xyz = q[1:]
    return (
        vector
        + 2.0 * w * np.cross(xyz, vector)
        + 2.0 * np.cross(xyz, np.cross(xyz, vector))
    )


def _quat_pitch(q: np.ndarray) -> float:
    sin_pitch = 2.0 * (float(q[0]) * float(q[2]) - float(q[3]) * float(q[1]))
    return float(np.arcsin(_clamp(sin_pitch, -1.0, 1.0)))


@dataclass(frozen=True)
class _Spline:
    x: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray

    def evaluate(self, coordinate: float) -> tuple[float, float, float]:
        segment = 0
        while segment + 1 < len(self.x) - 1 and coordinate > self.x[segment + 1]:
            segment += 1
        delta = coordinate - float(self.x[segment])
        value = (
            self.a[segment]
            + self.b[segment] * delta
            + self.c[segment] * delta * delta
            + self.d[segment] * delta * delta * delta
        )
        first = (
            self.b[segment]
            + 2.0 * self.c[segment] * delta
            + 3.0 * self.d[segment] * delta * delta
        )
        second = 2.0 * self.c[segment] + 6.0 * self.d[segment] * delta
        return float(value), float(first), float(second)


def _natural_spline(x: np.ndarray, values: np.ndarray) -> _Spline:
    points = len(x)
    h = np.diff(x)
    alpha = np.zeros(points, dtype=np.float64)
    alpha[1:-1] = 3.0 * (
        (values[2:] - values[1:-1]) / h[1:]
        - (values[1:-1] - values[:-2]) / h[:-1]
    )
    lower = np.zeros(points, dtype=np.float64)
    mu = np.zeros(points, dtype=np.float64)
    z = np.zeros(points, dtype=np.float64)
    knot_c = np.zeros(points, dtype=np.float64)
    lower[0] = 1.0
    for index in range(1, points - 1):
        lower[index] = (
            2.0 * (x[index + 1] - x[index - 1])
            - h[index - 1] * mu[index - 1]
        )
        mu[index] = h[index] / lower[index]
        z[index] = (alpha[index] - h[index - 1] * z[index - 1]) / lower[index]
    lower[-1] = 1.0
    a = values[:-1].copy()
    b = np.zeros(points - 1, dtype=np.float64)
    c = np.zeros(points - 1, dtype=np.float64)
    d = np.zeros(points - 1, dtype=np.float64)
    for index in range(points - 2, -1, -1):
        knot_c[index] = z[index] - mu[index] * knot_c[index + 1]
        b[index] = (
            (values[index + 1] - values[index]) / h[index]
            - h[index] * (knot_c[index + 1] + 2.0 * knot_c[index]) / 3.0
        )
        c[index] = knot_c[index]
        d[index] = (knot_c[index + 1] - knot_c[index]) / (3.0 * h[index])
    return _Spline(x.copy(), a, b, c, d)


def _exit_spline(x: np.ndarray, values: np.ndarray) -> _Spline:
    natural = _natural_spline(x, values)
    points = len(x)
    slopes = np.zeros(points, dtype=np.float64)
    exit_tangent_from_point = 3
    for index in range(points - 1):
        if index >= exit_tangent_from_point:
            slopes[index] = (values[index + 1] - values[index]) / (
                x[index + 1] - x[index]
            )
        else:
            slopes[index] = natural.b[index]
    slopes[-1] = slopes[-2]
    h = np.diff(x)
    secant = np.diff(values) / h
    a = values[:-1].copy()
    b = slopes[:-1].copy()
    c = (3.0 * secant - 2.0 * slopes[:-1] - slopes[1:]) / h
    d = (slopes[:-1] + slopes[1:] - 2.0 * secant) / (h * h)
    return _Spline(x.copy(), a, b, c, d)


_COURSE = np.vstack((COURSE_ORIGIN, GATES))
_LATERAL_SPLINE = _exit_spline(_COURSE[:, 0], _COURSE[:, 1])
_VERTICAL_SPLINE = _exit_spline(_COURSE[:, 0], _COURSE[:, 2])


class ObservableTailController:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.previous_rel_world = np.zeros(3, dtype=np.float64)
        self.velocity_world = np.zeros(3, dtype=np.float64)
        self.previous_gate = -1
        self.pose_valid = False
        self.velocity_valid = False

    def act(self, observation: np.ndarray, policy_action: Sequence[float]) -> list[float]:
        gate = min(max(six_gate._gate_index(observation), 0), NUM_GATES - 1)
        q = np.asarray(observation[6:10], dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm > 1e-6:
            q /= norm
        else:
            q[:] = (1.0, 0.0, 0.0, 0.0)

        rel_body = np.asarray(
            (
                _unscale_tanh(observation[11], 0.1),
                _unscale_tanh(observation[12], 0.2),
                -_unscale_tanh(observation[13], 0.2),
            ),
            dtype=np.float64,
        )
        rel_world = _quat_rotate(q, rel_body)
        gate_rate_body = np.asarray(
            (
                _unscale_tanh(observation[0], 0.2),
                _unscale_tanh(observation[1], 1.0 / 3.0),
                -_unscale_tanh(observation[2], 1.0 / 3.0),
            ),
            dtype=np.float64,
        )
        motion_velocity = -_quat_rotate(q, gate_rate_body)

        if not self.velocity_valid:
            self.velocity_world = motion_velocity.copy()
            self.velocity_valid = True
        if self.pose_valid and self.previous_gate == gate:
            rel_delta = rel_world - self.previous_rel_world
            displacement = float(np.linalg.norm(rel_delta))
            if 1e-5 < displacement < 1.0:
                finite_difference_velocity = -rel_delta / DT
                self.velocity_world += 0.35 * (
                    finite_difference_velocity - self.velocity_world
                )
        velocity_world = 0.5 * self.velocity_world + 0.5 * motion_velocity
        self.previous_rel_world = rel_world.copy()
        self.previous_gate = gate
        self.pose_valid = True

        course_position = GATES[gate] - rel_world
        desired_y, desired_dy_dx, desired_d2y_dx2 = _LATERAL_SPLINE.evaluate(
            float(course_position[0])
        )
        desired_z, desired_dz_dx, desired_d2z_dx2 = _VERTICAL_SPLINE.evaluate(
            float(course_position[0])
        )
        if gate >= 2:
            lookahead_x = min(
                float(course_position[0]) + 6.10,
                float(_LATERAL_SPLINE.x[-1]),
            )
            _, desired_dy_dx, desired_d2y_dx2 = _LATERAL_SPLINE.evaluate(lookahead_x)

        forward_speed = max(float(velocity_world[0]), 0.0)
        close_blend = _clamp(1.0 - max(float(rel_world[0]), 0.0) / 10.0, 0.0, 1.0)
        lateral_kp = 1.5 + 2.5 * close_blend
        accel_y = (
            desired_d2y_dx2 * forward_speed * forward_speed
            + lateral_kp * (desired_y - float(course_position[1]))
            + 2.7 * (desired_dy_dx * forward_speed - float(velocity_world[1]))
        )
        accel_z = (
            desired_d2z_dx2 * forward_speed * forward_speed
            + 2.2 * (desired_z - float(course_position[2]))
            + 2.0 * (desired_dz_dx * forward_speed - float(velocity_world[2]))
        )
        force_y = accel_y + 0.14 * float(velocity_world[1])
        force_z = 8.69 + accel_z + 0.14 * float(velocity_world[2])
        pitch = _quat_pitch(q)
        desired_roll = atan2(
            -force_y * cos(pitch),
            max(force_z * 0.107491, 1e-3),
        )
        desired_roll = _clamp(desired_roll, -0.50, 0.50)
        roll_action = desired_roll / 0.50
        # Match the native gate-local adapter: do not keep integrating the
        # counter-bias once the observed gate plane is behind the vehicle.
        if float(observation[17]) >= 0.0 and float(observation[11]) >= 0.0:
            roll_action += float(ROLL_BIAS[gate])

        vertical_fraction = max(cos(desired_roll) * cos(pitch), 0.50)
        command_thrust = force_z / (32.81 * vertical_fraction)
        if command_thrust >= 0.27:
            thrust_action = (command_thrust - 0.27) / (0.42 - 0.27)
        else:
            thrust_action = (command_thrust - 0.27) / (0.27 - 0.18)

        action = [float(value) for value in policy_action]
        action[1] = _clamp(roll_action, -1.0, 1.0)
        action[2] = _clamp(thrust_action, -1.0, 1.0)
        return action


_CONTROLLER = ObservableTailController()
_CONTROLLER_ACTIVE = False


def reset() -> None:
    global _CONTROLLER_ACTIVE
    n219.reset()
    _CONTROLLER.reset()
    _CONTROLLER_ACTIVE = False


def infer(observation: Sequence[float]) -> list[float]:
    global _CONTROLLER_ACTIVE
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    policy_action = n219.infer(values)
    gate = six_gate._gate_index(values)
    if gate <= 2:
        if _CONTROLLER_ACTIVE:
            _CONTROLLER.reset()
        _CONTROLLER_ACTIVE = False
        return policy_action
    if not _CONTROLLER_ACTIVE:
        _CONTROLLER.reset()
        _CONTROLLER_ACTIVE = True
    return _CONTROLLER.act(values, policy_action)


def controller_snapshot() -> dict:
    return {
        "n220_observable_tail": {
            "active": _CONTROLLER_ACTIVE,
            "previous_gate": _CONTROLLER.previous_gate,
            "parameters": {
                "prefix_gate_indices": [0, 1, 2],
                "observable_controller_gate_indices": [3, 4, 5],
                "policy_owned_actions": ["pitch", "yaw"],
                "controller_owned_actions": ["roll", "thrust"],
                "runtime_privileged_state": False,
                "motion_rate_blend": 0.5,
                "gate_roll_bias": ROLL_BIAS.tolist(),
                "control_aware_gate_predictor": True,
            },
        },
        "n219": n219.controller_snapshot(),
    }
