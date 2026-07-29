#!/usr/bin/env python3
"""Bounded, uncertainty-aware visual receding-horizon control primitives.

The controller consumes only the deployed 32-value observation.  It carries a
world-frame velocity estimate across the Gate-3/Gate-4 transition, associates
current raw Gate-4 poses for at most a bounded dropout interval, and optimizes
short desired-pitch/roll/thrust sequences against a leave-one-flight-fold-out
dynamics ensemble.  Only the first control is returned; every accepted camera
pose causes a replan.

There is intentionally no unbounded extrapolation.  The optimizer terminates
each simulated member at the gate plane or a two-second horizon, rejects plans
whose ensemble disagreement is too large, and emits a level/braking failsafe
when current visual state is unavailable.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np


GYRO_SCALE_RAD_S = 20.0
MAX_ROLL_RAD = 0.5
MAX_PITCH_RAD = 0.5
MAX_YAW_RAD = math.pi
HOVER_THRUST = 0.27
MIN_THRUST = 0.18
MAX_THRUST = 0.42


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def inverse_tanh(value: float, scale: float) -> float:
    return float(np.arctanh(np.clip(value, -0.999999, 0.999999)) * scale)


def quaternion_rotate(
    quaternion_wxyz: Sequence[float], vector: Sequence[float]
) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    value = np.asarray(vector, dtype=np.float64)
    xyz = quaternion[1:]
    return value + 2.0 * (
        quaternion[0] * np.cross(xyz, value)
        + np.cross(xyz, np.cross(xyz, value))
    )


def quaternion_inverse_rotate(
    quaternion_wxyz: Sequence[float], vector: Sequence[float]
) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).copy()
    quaternion[1:] *= -1.0
    return quaternion_rotate(quaternion, vector)


def quaternion_to_euler(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return np.asarray(
        [
            math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            math.asin(clamp(2.0 * (w * y - z * x), -1.0, 1.0)),
            math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ],
        dtype=np.float64,
    )


def _euler_to_quaternion_batch(euler: np.ndarray) -> np.ndarray:
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=-1,
    )


def _quaternion_rotate_batch(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[..., 1:]
    return vector + 2.0 * (
        quaternion[..., :1] * np.cross(xyz, vector)
        + np.cross(xyz, np.cross(xyz, vector))
    )


def _quaternion_inverse_rotate_batch(
    quaternion: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    inverse = quaternion.copy()
    inverse[..., 1:] *= -1.0
    return _quaternion_rotate_batch(inverse, vector)


def command_thrust(normalized: np.ndarray | float) -> np.ndarray:
    action = np.clip(np.asarray(normalized, dtype=np.float64), -1.0, 1.0)
    positive = HOVER_THRUST + action * (MAX_THRUST - HOVER_THRUST)
    negative = HOVER_THRUST + action * (HOVER_THRUST - MIN_THRUST)
    return np.where(action >= 0.0, positive, negative)


@dataclasses.dataclass(frozen=True)
class VisualDynamicsEnsemble:
    thrust_gain: np.ndarray
    linear_drag_per_s: np.ndarray
    acceleration_bias_m_s2: np.ndarray
    angle_error_gain_per_s2: np.ndarray
    omega_damping_per_s: np.ndarray
    omega_bias_rad_s2: np.ndarray
    source_path: str
    schema: str

    @property
    def members(self) -> int:
        return int(self.thrust_gain.shape[0])

    @classmethod
    def load(cls, path: str | Path) -> "VisualDynamicsEnsemble":
        resolved = Path(path).resolve()
        report = json.loads(resolved.read_text(encoding="utf-8"))
        if report.get("schema") != "official_visual_dynamics_v1":
            raise ValueError("unsupported visual dynamics schema")
        models = report.get("fold_models") or []
        if len(models) < 2:
            raise ValueError("visual dynamics report needs at least two fold models")

        def arrays(section: str, key: str) -> np.ndarray:
            return np.asarray(
                [
                    [float(axis[key]) for axis in model[section]["axes"]]
                    for model in models
                ],
                dtype=np.float64,
            )

        output = cls(
            thrust_gain=arrays("translation", "thrust_gain"),
            linear_drag_per_s=arrays("translation", "linear_drag_per_s"),
            acceleration_bias_m_s2=arrays("translation", "bias_m_s2"),
            angle_error_gain_per_s2=arrays("attitude", "angle_error_gain_per_s2"),
            omega_damping_per_s=arrays("attitude", "omega_damping_per_s"),
            omega_bias_rad_s2=arrays("attitude", "bias_rad_s2"),
            source_path=str(resolved),
            schema=str(report["schema"]),
        )
        for name, value in dataclasses.asdict(output).items():
            if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite model coefficient: {name}")
        if np.any(output.thrust_gain < 0.0) or np.any(output.linear_drag_per_s < 0.0):
            raise ValueError("deployment model violates non-negative plant constraints")
        if np.any(output.omega_damping_per_s <= 0.0):
            raise ValueError("deployment model has unstable attitude damping")
        return output


@dataclasses.dataclass
class VisualState:
    relative_world_m: np.ndarray
    velocity_world_m_s: np.ndarray
    euler_rpy_rad: np.ndarray
    omega_rpy_rad_s: np.ndarray
    body_ned_m: np.ndarray
    reference_quaternion_wxyz: np.ndarray
    reference_ned_m: np.ndarray
    position_uncertainty_m: float
    measurement_age_s: float

    def copy(self) -> "VisualState":
        return VisualState(
            relative_world_m=self.relative_world_m.copy(),
            velocity_world_m_s=self.velocity_world_m_s.copy(),
            euler_rpy_rad=self.euler_rpy_rad.copy(),
            omega_rpy_rad_s=self.omega_rpy_rad_s.copy(),
            body_ned_m=self.body_ned_m.copy(),
            reference_quaternion_wxyz=self.reference_quaternion_wxyz.copy(),
            reference_ned_m=self.reference_ned_m.copy(),
            position_uncertainty_m=float(self.position_uncertainty_m),
            measurement_age_s=float(self.measurement_age_s),
        )


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    maximum_horizon_s: float = 2.0
    minimum_horizon_s: float = 0.8
    integration_dt_s: float = 0.05
    control_knots: int = 4
    candidates: int = 32
    iterations: int = 2
    elite_fraction: float = 0.12
    uncertainty_limit_m: float = 0.80
    plane_forward_m: float = 0.15
    terminal_center_limit_m: float = 0.75
    maximum_pitch_norm: float = 1.00
    minimum_pitch_norm: float = -0.30
    maximum_roll_norm: float = 0.90
    risk_standard_deviations: float = 1.5
    worst_case_weight: float = 1.0
    lateral_projection_horizon_s: float = 2.40


@dataclasses.dataclass(frozen=True)
class PlanResult:
    action: tuple[float, float, float, float]
    accepted: bool
    reason: str
    horizon_s: float
    objective: float | None
    ensemble_disagreement_m: float | None
    predicted_terminal_body_ned_m: tuple[float, float, float] | None
    crossing_members: int


class Gate4RecedingHorizonOptimizer:
    """Deterministic risk-sensitive random-shooting optimizer."""

    def __init__(
        self,
        ensemble: VisualDynamicsEnsemble,
        config: OptimizerConfig | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.config = config or OptimizerConfig()
        self.plan_count = 0

    def reset(self) -> None:
        self.plan_count = 0

    def _failsafe_action(
        self, state: VisualState, yaw_target_rad: float
    ) -> tuple[float, float, float, float]:
        # Positive desired pitch is the measured braking direction for the
        # current forward axis.  A modest high thrust arrests the inherited
        # Gate-3 descent without saturating vertically.
        velocity_reference = quaternion_inverse_rotate(
            state.reference_quaternion_wxyz, state.velocity_world_m_s
        )
        velocity_reference[2] *= -1.0
        pitch_norm = clamp(
            0.12 * (float(velocity_reference[0]) - 4.0),
            0.0,
            1.0,
        )
        thrust_norm = clamp(
            -0.05 * float(state.reference_ned_m[2])
            + 0.30 * float(velocity_reference[2]),
            -0.4,
            1.0,
        )
        stopping_time_s = clamp(
            max(float(state.reference_ned_m[0]), 0.0)
            / max(float(velocity_reference[0]), 3.0),
            0.0,
            6.0,
        )
        lateral_control_error_m = float(state.reference_ned_m[1]) - float(
            velocity_reference[1]
        ) * min(
            stopping_time_s,
            self.config.lateral_projection_horizon_s,
        )
        roll_norm = clamp(
            -0.25 * lateral_control_error_m,
            -0.90,
            0.90,
        )
        projected_down_m = float(state.reference_ned_m[2]) - float(
            velocity_reference[2]
        ) * stopping_time_s
        projected_right_m = float(state.reference_ned_m[1]) - float(
            velocity_reference[1]
        ) * stopping_time_s
        if projected_down_m < -0.50:
            thrust_norm = max(
                thrust_norm, clamp(-0.12 * projected_down_m, 0.0, 1.0)
            )
        return (
            pitch_norm,
            roll_norm,
            thrust_norm,
            clamp(yaw_target_rad / math.pi),
        )

    def _horizon(self, state: VisualState) -> float:
        forward_m = max(float(state.reference_ned_m[0]), 0.1)
        velocity_body = _quaternion_inverse_rotate_batch(
            state.reference_quaternion_wxyz[None, :],
            state.velocity_world_m_s[None, :],
        )[0]
        closing_m_s = max(float(velocity_body[0]), 3.0)
        time_to_plane = forward_m / closing_m_s
        return clamp(
            time_to_plane + 0.25,
            self.config.minimum_horizon_s,
            self.config.maximum_horizon_s,
        )

    def _initial_mean(self, state: VisualState) -> np.ndarray:
        cfg = self.config
        forward, right, down = (float(value) for value in state.reference_ned_m)
        velocity_body = _quaternion_inverse_rotate_batch(
            state.reference_quaternion_wxyz[None, :],
            state.velocity_world_m_s[None, :],
        )[0]
        velocity_body[2] *= -1.0
        closing = float(velocity_body[0])
        # The forward fit is intentionally weakly trusted: cap pitch to a
        # narrow braking/level envelope and let measured inertia do the work.
        pitch = clamp(0.06 * (closing - 6.0), cfg.minimum_pitch_norm, cfg.maximum_pitch_norm)
        roll = clamp(-0.12 * right - 0.05 * float(velocity_body[1]), -0.8, 0.8)
        # Upward thrust must brake negative world-Z velocity inherited from
        # Gate 3; the optimizer then schedules it down as the opening centers.
        thrust = clamp(
            -0.05 * down + 0.30 * float(velocity_body[2]),
            -0.6,
            1.0,
        )
        terminal = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        first = np.asarray([pitch, roll, thrust], dtype=np.float64)
        return np.stack(
            [
                first + (terminal - first) * (index / max(cfg.control_knots - 1, 1))
                for index in range(cfg.control_knots)
            ],
            axis=0,
        )

    def _simulate(
        self,
        state: VisualState,
        controls: np.ndarray,
        *,
        yaw_target_rad: float,
        horizon_s: float,
    ) -> tuple[np.ndarray, dict]:
        cfg = self.config
        candidates = controls.shape[0]
        members = self.ensemble.members
        shape = (candidates, members, 3)
        relative = np.broadcast_to(state.relative_world_m, shape).copy()
        velocity = np.broadcast_to(state.velocity_world_m_s, shape).copy()
        euler = np.broadcast_to(state.euler_rpy_rad, shape).copy()
        omega = np.broadcast_to(state.omega_rpy_rad_s, shape).copy()
        active = np.ones((candidates, members), dtype=bool)
        crossed = np.zeros((candidates, members), dtype=bool)
        costs = np.zeros((candidates, members), dtype=np.float64)
        terminal_body = np.full(shape, np.nan, dtype=np.float64)
        last_body = np.broadcast_to(state.reference_ned_m, shape).copy()
        initial_forward = max(float(state.reference_ned_m[0]), 1.0)
        initial_right = float(state.reference_ned_m[1])
        initial_down = float(state.reference_ned_m[2])
        reference_quaternion = np.broadcast_to(
            state.reference_quaternion_wxyz, shape[:-1] + (4,)
        )
        steps = max(1, int(math.ceil(horizon_s / cfg.integration_dt_s)))
        dt = horizon_s / steps
        knot_duration = horizon_s / cfg.control_knots
        model_axis = (None, slice(None), slice(None))
        thrust_gain = self.ensemble.thrust_gain[model_axis]
        drag = self.ensemble.linear_drag_per_s[model_axis]
        accel_bias = self.ensemble.acceleration_bias_m_s2[model_axis]
        angle_gain = self.ensemble.angle_error_gain_per_s2[model_axis]
        damping = self.ensemble.omega_damping_per_s[model_axis]
        omega_bias = self.ensemble.omega_bias_rad_s2[model_axis]

        for step_index in range(steps):
            elapsed = step_index * dt
            knot = min(int(elapsed / max(knot_duration, 1e-6)), cfg.control_knots - 1)
            command = controls[:, knot, :]
            desired = np.empty((candidates, 1, 3), dtype=np.float64)
            desired[:, 0, 0] = command[:, 1] * MAX_ROLL_RAD
            desired[:, 0, 1] = command[:, 0] * MAX_PITCH_RAD
            desired[:, 0, 2] = yaw_target_rad
            angle_error = desired - euler
            angle_error[..., 2] = np.arctan2(
                np.sin(angle_error[..., 2]), np.cos(angle_error[..., 2])
            )
            omega += (angle_gain * angle_error - damping * omega + omega_bias) * dt * active[..., None]
            euler += omega * dt * active[..., None]
            euler[..., 2] = np.arctan2(np.sin(euler[..., 2]), np.cos(euler[..., 2]))
            quaternion = _euler_to_quaternion_batch(euler)
            up = _quaternion_rotate_batch(
                quaternion, np.broadcast_to(np.asarray((0.0, 0.0, 1.0)), shape)
            )
            thrust_axis = up * command_thrust(command[:, None, 2])[..., None]
            thrust_axis[..., 0] *= -1.0
            acceleration = thrust_gain * thrust_axis - drag * velocity + accel_bias
            velocity += acceleration * dt * active[..., None]
            relative -= velocity * dt * active[..., None]
            # Centering and plane termination use the inertial frame anchored
            # on the first coherent Gate-4 pose.  Using the changing body frame
            # would let a candidate appear centered merely by banking/yawing
            # the camera without moving toward the aperture center.
            body_up = _quaternion_inverse_rotate_batch(
                reference_quaternion, relative
            )
            body = body_up.copy()
            body[..., 2] *= -1.0
            last_body = body
            ratio = np.clip(body[..., 0] / initial_forward, 0.0, 1.0)
            desired_right = initial_right * np.power(ratio, 1.35)
            desired_down = initial_down * np.power(ratio, 1.35)
            center_error = np.square(body[..., 1] - desired_right) + 1.25 * np.square(
                body[..., 2] - desired_down
            )
            proximity = 1.0 + 4.0 * np.square(1.0 - ratio)
            costs += active * dt * proximity * center_error
            costs += active * dt * (
                0.03 * np.square(command[:, None, 0])
                + 0.05 * np.square(command[:, None, 1])
                + 0.02 * np.square(command[:, None, 2])
            )
            newly_crossed = active & (body[..., 0] <= cfg.plane_forward_m)
            if np.any(newly_crossed):
                terminal_body[newly_crossed] = body[newly_crossed]
                terminal_radius_sq = np.square(body[..., 1]) + np.square(body[..., 2])
                velocity_body = _quaternion_inverse_rotate_batch(
                    reference_quaternion, velocity
                )
                velocity_body[..., 2] *= -1.0
                lateral_speed_sq = np.square(velocity_body[..., 1]) + np.square(
                    velocity_body[..., 2]
                )
                miss = (
                    np.abs(body[..., 1]) > cfg.terminal_center_limit_m
                ) | (np.abs(body[..., 2]) > cfg.terminal_center_limit_m)
                costs += newly_crossed * (
                    180.0 * terminal_radius_sq
                    + 3.0 * lateral_speed_sq
                    + 900.0 * miss
                    + 0.5 * elapsed
                )
                crossed |= newly_crossed
                active &= ~newly_crossed

        # A non-crossing horizon is normal while the gate is still far away;
        # require progress along the shrinking centerline, not a fictional
        # terminal crossing beyond the model's time authority.
        ratio = np.clip(last_body[..., 0] / initial_forward, 0.0, 1.0)
        desired_right = initial_right * np.power(ratio, 1.35)
        desired_down = initial_down * np.power(ratio, 1.35)
        costs += active * (
            35.0 * np.square(last_body[..., 1] - desired_right)
            + 45.0 * np.square(last_body[..., 2] - desired_down)
            + 0.5 * np.square(np.maximum(last_body[..., 0], 0.0) / initial_forward)
        )
        member_mean = np.mean(costs, axis=1)
        member_std = np.std(costs, axis=1)
        objective = (
            member_mean
            + cfg.risk_standard_deviations * member_std
            + cfg.worst_case_weight * np.max(costs, axis=1)
        )
        return objective, {
            "costs": costs,
            "crossed": crossed,
            "terminal_body": terminal_body,
            "last_body": last_body,
        }

    def plan(self, state: VisualState, *, yaw_target_rad: float) -> PlanResult:
        cfg = self.config
        failsafe = self._failsafe_action(state, yaw_target_rad)
        if state.measurement_age_s > 0.30:
            return PlanResult(
                failsafe,
                False,
                "measurement_stale",
                0.0,
                None,
                None,
                None,
                0,
            )
        if state.position_uncertainty_m > cfg.uncertainty_limit_m:
            return PlanResult(
                failsafe,
                False,
                "state_uncertain",
                0.0,
                None,
                state.position_uncertainty_m,
                None,
                0,
            )
        horizon = self._horizon(state)
        velocity_reference = quaternion_inverse_rotate(
            state.reference_quaternion_wxyz, state.velocity_world_m_s
        )
        velocity_reference[2] *= -1.0
        forward_m, right_m, down_m = (
            float(value) for value in state.reference_ned_m
        )
        closing_m_s = max(float(velocity_reference[0]), 3.0)
        time_to_plane_s = max(forward_m, 0.0) / closing_m_s
        projected_right_m = right_m - float(velocity_reference[1]) * time_to_plane_s
        projected_down_m = down_m - float(velocity_reference[2]) * time_to_plane_s
        center_error_m = max(abs(projected_right_m), abs(projected_down_m))
        pitch_brake_floor = (
            clamp(
                0.12 * (closing_m_s - 4.0),
                0.0,
                cfg.maximum_pitch_norm,
            )
            if center_error_m > cfg.terminal_center_limit_m
            else None
        )
        vertical_brake_floor = (
            clamp(-0.12 * projected_down_m, 0.0, 1.0)
            if projected_down_m < -0.50
            else None
        )
        lateral_first_bound: float | None = None
        # A full time-to-plane linear projection counter-banked too early on
        # N230, while current error alone counter-banked too late.  Bound the
        # stopping lookahead to the validated short-horizon authority.
        lateral_control_error_m = right_m - float(velocity_reference[1]) * min(
            time_to_plane_s,
            cfg.lateral_projection_horizon_s,
        )
        if lateral_control_error_m > 0.50:
            lateral_first_bound = clamp(
                -0.25 * lateral_control_error_m,
                -cfg.maximum_roll_norm,
                -0.15,
            )
        elif lateral_control_error_m < -0.50:
            lateral_first_bound = clamp(
                -0.25 * lateral_control_error_m,
                0.15,
                cfg.maximum_roll_norm,
            )
        mean = self._initial_mean(state)
        sigma = np.broadcast_to(
            np.asarray((0.12, 0.30, 0.35), dtype=np.float64), mean.shape
        ).copy()
        rng = np.random.default_rng(232000 + self.plan_count)
        self.plan_count += 1
        elite_count = max(4, int(round(cfg.candidates * cfg.elite_fraction)))
        best_controls = mean[None, ...]
        best_objective = math.inf
        best_details: dict | None = None
        best_index = 0
        for _ in range(cfg.iterations):
            controls = rng.normal(
                loc=mean[None, ...],
                scale=sigma[None, ...],
                size=(cfg.candidates, cfg.control_knots, 3),
            )
            controls[0] = mean
            controls[..., 0] = np.clip(
                controls[..., 0], cfg.minimum_pitch_norm, cfg.maximum_pitch_norm
            )
            controls[..., 1] = np.clip(
                controls[..., 1], -cfg.maximum_roll_norm, cfg.maximum_roll_norm
            )
            controls[..., 2] = np.clip(controls[..., 2], -1.0, 1.0)
            # The weakly identified forward model is never allowed to trade
            # away an observable lateral/vertical stopping requirement.  These
            # first-control bounds are recomputed from the current state on
            # every frame; subsequent knots remain optimizer-owned.
            if vertical_brake_floor is not None:
                controls[:, 0, 2] = np.maximum(
                    controls[:, 0, 2], vertical_brake_floor
                )
            if pitch_brake_floor is not None:
                controls[:, 0, 0] = np.maximum(
                    controls[:, 0, 0], pitch_brake_floor
                )
            if lateral_first_bound is not None:
                if lateral_control_error_m > 0.0:
                    controls[:, 0, 1] = np.minimum(
                        controls[:, 0, 1], lateral_first_bound
                    )
                else:
                    controls[:, 0, 1] = np.maximum(
                        controls[:, 0, 1], lateral_first_bound
                    )
            objective, details = self._simulate(
                state, controls, yaw_target_rad=yaw_target_rad, horizon_s=horizon
            )
            order = np.argsort(objective)
            elites = controls[order[:elite_count]]
            mean = np.mean(elites, axis=0)
            sigma = np.maximum(np.std(elites, axis=0), np.asarray((0.025, 0.04, 0.05)))
            if float(objective[order[0]]) < best_objective:
                best_objective = float(objective[order[0]])
                best_controls = controls
                best_details = details
                best_index = int(order[0])

        assert best_details is not None
        crossed = best_details["crossed"][best_index]
        crossing_members = int(np.sum(crossed))
        positions = np.where(
            crossed[:, None],
            best_details["terminal_body"][best_index],
            best_details["last_body"][best_index],
        )
        lateral_vertical = positions[:, 1:3]
        disagreement = float(
            np.max(np.linalg.norm(lateral_vertical - np.mean(lateral_vertical, axis=0), axis=1))
        )
        if disagreement > cfg.uncertainty_limit_m:
            return PlanResult(
                failsafe,
                False,
                "prediction_uncertain",
                horizon,
                best_objective,
                disagreement,
                None,
                crossing_members,
            )
        first = best_controls[best_index, 0]
        predicted = np.mean(positions, axis=0)
        return PlanResult(
            (
                clamp(first[0]),
                clamp(first[1]),
                clamp(first[2]),
                clamp(yaw_target_rad / math.pi),
            ),
            True,
            "optimized",
            horizon,
            best_objective,
            disagreement,
            tuple(float(value) for value in predicted),
            crossing_members,
        )


def propagate_member(
    state: VisualState,
    action: Sequence[float],
    ensemble: VisualDynamicsEnsemble,
    *,
    member: int,
    duration_s: float,
    integration_dt_s: float = 0.02,
) -> VisualState:
    """Propagate one held-out ensemble member for counterfactual validation."""

    if not 0 <= member < ensemble.members:
        raise IndexError("ensemble member out of range")
    normalized = np.asarray(action, dtype=np.float64)
    if normalized.shape != (4,):
        raise ValueError("expected four normalized actions")
    output = state.copy()
    steps = max(1, int(math.ceil(duration_s / integration_dt_s)))
    dt = float(duration_s) / steps
    desired = np.asarray(
        (
            clamp(normalized[1]) * MAX_ROLL_RAD,
            clamp(normalized[0]) * MAX_PITCH_RAD,
            clamp(normalized[3]) * MAX_YAW_RAD,
        ),
        dtype=np.float64,
    )
    for _ in range(steps):
        error = np.asarray(
            [wrap_angle(desired[index] - output.euler_rpy_rad[index]) for index in range(3)]
        )
        output.omega_rpy_rad_s += (
            ensemble.angle_error_gain_per_s2[member] * error
            - ensemble.omega_damping_per_s[member] * output.omega_rpy_rad_s
            + ensemble.omega_bias_rad_s2[member]
        ) * dt
        output.euler_rpy_rad += output.omega_rpy_rad_s * dt
        output.euler_rpy_rad[2] = wrap_angle(output.euler_rpy_rad[2])
        quaternion = _euler_to_quaternion_batch(output.euler_rpy_rad[None, :])[0]
        up = _quaternion_rotate_batch(
            quaternion[None, :], np.asarray(((0.0, 0.0, 1.0),))
        )[0]
        thrust_axis = up * float(command_thrust(normalized[2]))
        thrust_axis[0] *= -1.0
        acceleration = (
            ensemble.thrust_gain[member] * thrust_axis
            - ensemble.linear_drag_per_s[member] * output.velocity_world_m_s
            + ensemble.acceleration_bias_m_s2[member]
        )
        output.velocity_world_m_s += acceleration * dt
        output.relative_world_m -= output.velocity_world_m_s * dt
    quaternion = _euler_to_quaternion_batch(output.euler_rpy_rad[None, :])[0]
    body_up = _quaternion_inverse_rotate_batch(
        quaternion[None, :], output.relative_world_m[None, :]
    )[0]
    output.body_ned_m = body_up
    output.body_ned_m[2] *= -1.0
    reference_up = quaternion_inverse_rotate(
        output.reference_quaternion_wxyz, output.relative_world_m
    )
    output.reference_ned_m = reference_up
    output.reference_ned_m[2] *= -1.0
    output.measurement_age_s = 0.0
    return output


@dataclasses.dataclass(frozen=True)
class TrackerConfig:
    maximum_initial_range_m: float = 35.0
    maximum_innovation_base_m: float = 3.0
    maximum_innovation_speed_m_s: float = 30.0
    maximum_sample_gap_s: float = 0.50
    position_tau_s: float = 0.22
    velocity_tau_s: float = 0.40
    maximum_prediction_age_s: float = 0.30
    initial_position_uncertainty_m: float = 0.55
    maximum_position_uncertainty_m: float = 2.0


class Gate4VisualStateTracker:
    """Observable state tracker with bounded Gate-3 velocity carryover."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.reset()

    def reset(self) -> None:
        self.gate4_active = False
        self.carried_velocity_world_m_s: np.ndarray | None = None
        self.raw_world_m: np.ndarray | None = None
        self.filtered_world_m: np.ndarray | None = None
        self.velocity_world_m_s: np.ndarray | None = None
        self.reference_quaternion_wxyz: np.ndarray | None = None
        self.last_pose_key: tuple[float, float, float] | None = None
        self.last_update_s: float | None = None
        self.last_measurement_s: float | None = None
        self.position_uncertainty_m = self.config.initial_position_uncertainty_m
        self.accepted_samples = 0
        self.rejected_samples = 0

    @staticmethod
    def _quaternion(values: np.ndarray) -> np.ndarray:
        return np.asarray(values[6:10], dtype=np.float64)

    @staticmethod
    def _body_pose(values: np.ndarray) -> np.ndarray | None:
        if float(values[10]) < 0.5:
            return None
        return np.asarray(
            [
                inverse_tanh(float(values[11]), 10.0),
                inverse_tanh(float(values[12]), 5.0),
                inverse_tanh(float(values[13]), 5.0),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _vehicle_velocity_from_gate_rate(values: np.ndarray) -> np.ndarray | None:
        if float(values[10]) < 0.5:
            return None
        relative_body_up = np.asarray(
            [
                inverse_tanh(float(values[0]), 5.0),
                inverse_tanh(float(values[1]), 3.0),
                -inverse_tanh(float(values[2]), 3.0),
            ],
            dtype=np.float64,
        )
        relative_world = quaternion_rotate(values[6:10], relative_body_up)
        velocity = -relative_world
        if not np.all(np.isfinite(velocity)) or np.linalg.norm(velocity) > 25.0:
            return None
        return velocity

    def observe_prefix(self, values: Sequence[float], *, gate_index: int) -> None:
        array = np.asarray(values, dtype=np.float64)
        if gate_index != 2:
            return
        velocity = self._vehicle_velocity_from_gate_rate(array)
        if velocity is not None:
            self.carried_velocity_world_m_s = velocity

    def leave_gate4(self) -> None:
        carried = self.carried_velocity_world_m_s
        self.reset()
        self.carried_velocity_world_m_s = carried

    def update_gate4(
        self,
        values: Sequence[float],
        *,
        now_s: float | None = None,
    ) -> tuple[VisualState | None, bool]:
        array = np.asarray(values, dtype=np.float64)
        now = time.monotonic() if now_s is None else float(now_s)
        quaternion = self._quaternion(array)
        euler = quaternion_to_euler(quaternion)
        omega = np.asarray(array[3:6], dtype=np.float64) * GYRO_SCALE_RAD_S
        body = self._body_pose(array)
        fresh = False
        if body is not None:
            pose_key = tuple(float(value) for value in body)
            range_m = float(np.linalg.norm(body))
            if self.raw_world_m is None and range_m > self.config.maximum_initial_range_m:
                self.last_pose_key = pose_key
                self.rejected_samples += 1
            elif pose_key != self.last_pose_key:
                world = quaternion_rotate(quaternion, (body[0], body[1], -body[2]))
                if self.raw_world_m is None:
                    self.raw_world_m = world
                    self.filtered_world_m = world.copy()
                    self.velocity_world_m_s = (
                        np.zeros(3, dtype=np.float64)
                        if self.carried_velocity_world_m_s is None
                        else self.carried_velocity_world_m_s.copy()
                    )
                    self.reference_quaternion_wxyz = quaternion.copy()
                    self.last_update_s = now
                    self.last_measurement_s = now
                    self.last_pose_key = pose_key
                    self.accepted_samples += 1
                    self.gate4_active = True
                    fresh = True
                else:
                    assert self.filtered_world_m is not None
                    assert self.velocity_world_m_s is not None
                    assert self.last_update_s is not None
                    dt = now - self.last_update_s
                    if dt > 0.0:
                        association_dt = min(dt, self.config.maximum_sample_gap_s)
                        predicted = self.filtered_world_m - self.velocity_world_m_s * dt
                        innovation = world - predicted
                        innovation_norm = float(np.linalg.norm(innovation))
                        limit = max(
                            self.config.maximum_innovation_base_m,
                            self.config.maximum_innovation_speed_m_s * association_dt,
                        )
                        if innovation_norm <= limit:
                            old_filtered = self.filtered_world_m.copy()
                            position_alpha = 1.0 - math.exp(
                                -dt / max(self.config.position_tau_s, 1e-6)
                            )
                            self.filtered_world_m = predicted + position_alpha * innovation
                            measured_velocity = -(
                                self.filtered_world_m - old_filtered
                            ) / dt
                            velocity_alpha = 1.0 - math.exp(
                                -dt / max(self.config.velocity_tau_s, 1e-6)
                            )
                            self.velocity_world_m_s += velocity_alpha * (
                                measured_velocity - self.velocity_world_m_s
                            )
                            self.raw_world_m = world
                            self.last_update_s = now
                            self.last_measurement_s = now
                            self.last_pose_key = pose_key
                            self.position_uncertainty_m = clamp(
                                0.65 * self.position_uncertainty_m
                                + 0.20 * innovation_norm,
                                0.05,
                                self.config.maximum_position_uncertainty_m,
                            )
                            self.accepted_samples += 1
                            fresh = True
                        else:
                            self.last_pose_key = pose_key
                            self.position_uncertainty_m = min(
                                self.config.maximum_position_uncertainty_m,
                                self.position_uncertainty_m + 0.25,
                            )
                            self.rejected_samples += 1

        if self.filtered_world_m is None or self.velocity_world_m_s is None:
            return None, fresh
        assert self.last_measurement_s is not None
        age = max(0.0, now - self.last_measurement_s)
        bounded_age = min(age, self.config.maximum_prediction_age_s)
        predicted_world = self.filtered_world_m - self.velocity_world_m_s * bounded_age
        predicted_body_up = quaternion_inverse_rotate(quaternion, predicted_world)
        predicted_body = predicted_body_up.copy()
        predicted_body[2] *= -1.0
        assert self.reference_quaternion_wxyz is not None
        reference_up = quaternion_inverse_rotate(
            self.reference_quaternion_wxyz, predicted_world
        )
        reference_ned = reference_up.copy()
        reference_ned[2] *= -1.0
        uncertainty = min(
            self.config.maximum_position_uncertainty_m,
            self.position_uncertainty_m + 0.5 * age,
        )
        return (
            VisualState(
                relative_world_m=predicted_world,
                velocity_world_m_s=self.velocity_world_m_s.copy(),
                euler_rpy_rad=euler,
                omega_rpy_rad_s=omega,
                body_ned_m=predicted_body,
                reference_quaternion_wxyz=self.reference_quaternion_wxyz.copy(),
                reference_ned_m=reference_ned,
                position_uncertainty_m=uncertainty,
                measurement_age_s=age,
            ),
            fresh,
        )

    def snapshot(self) -> dict:
        return {
            "gate4_active": self.gate4_active,
            "accepted_samples": self.accepted_samples,
            "rejected_samples": self.rejected_samples,
            "position_uncertainty_m": self.position_uncertainty_m,
            "carried_velocity_world_m_s": None
            if self.carried_velocity_world_m_s is None
            else self.carried_velocity_world_m_s.tolist(),
            "filtered_world_m": None
            if self.filtered_world_m is None
            else self.filtered_world_m.tolist(),
            "velocity_world_m_s": None
            if self.velocity_world_m_s is None
            else self.velocity_world_m_s.tolist(),
            "reference_quaternion_wxyz": None
            if self.reference_quaternion_wxyz is None
            else self.reference_quaternion_wxyz.tolist(),
            "last_measurement_s": self.last_measurement_s,
            "config": dataclasses.asdict(self.config),
        }
