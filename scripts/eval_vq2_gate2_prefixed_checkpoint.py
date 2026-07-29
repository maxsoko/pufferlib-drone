#!/usr/bin/env python3
"""Evaluate Gate 2 after warming a Puffer policy with the official N295 trace.

This evaluator remains native-only and command-free.  It replays the legal
32-value observations recorded before N295's Gate-1 stop to reconstruct the
recurrent state that a continuation carries into Gate 2, then steps the native
SITL plant from the measured transition state with teacher blending fixed to
zero.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

try:
    from eval_drone_race_checkpoint import _add_derived_metrics, _load_config
    from policy_callable_checkpoint import CheckpointPolicy
    from train_full_policy_bc import SequencePufferNet
except ModuleNotFoundError:  # Imported as scripts.eval_vq2_gate2_prefixed_checkpoint.
    from scripts.eval_drone_race_checkpoint import _add_derived_metrics, _load_config
    from scripts.policy_callable_checkpoint import CheckpointPolicy
    from scripts.train_full_policy_bc import SequencePufferNet


OBSERVATIONS = 32
ACTIONS = 4
LAST_ACTION_START = 19
START_PERTURBATION_DIMENSIONS = (
    "elapsed", "position", "velocity", "attitude", "rates")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_prefix(
    report_path: Path,
    *,
    stop_at_gate_advance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    samples = report.get("policy_trace", {}).get("samples", [])
    if not samples:
        raise ValueError("official report contains no policy_trace.samples")
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        observation = np.asarray(sample.get("observation"), dtype=np.float32)
        action = np.asarray(sample.get("normalized_action"), dtype=np.float32)
        if observation.shape != (OBSERVATIONS,):
            raise ValueError(f"prefix sample {index} has invalid observation width")
        if action.shape != (ACTIONS,):
            raise ValueError(f"prefix sample {index} has invalid action width")
        if int(sample.get("official_active_gate_index", -1)) != 0:
            if stop_at_gate_advance:
                break
            raise ValueError("official prefix extends beyond Gate 1")
        observations.append(observation)
        actions.append(action)
    if not observations:
        raise ValueError("official report contains no Gate-1 prefix samples")
    return np.stack(observations), np.stack(actions)


def gate2_first_observation(
    native_observations: np.ndarray,
    final_prefix_action: np.ndarray,
    initial_gate_rates: np.ndarray | None = None,
) -> np.ndarray:
    result = np.asarray(native_observations, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != OBSERVATIONS:
        raise ValueError("native observations must have shape [agents, 32]")
    action = np.asarray(final_prefix_action, dtype=np.float32)
    if action.shape != (ACTIONS,):
        raise ValueError("final prefix action must contain four values")
    result[:, LAST_ACTION_START:LAST_ACTION_START + ACTIONS] = action
    if initial_gate_rates is not None:
        rates = np.asarray(initial_gate_rates, dtype=np.float32)
        if rates.shape != (3,) or not np.isfinite(rates).all():
            raise ValueError("initial gate rates must contain three finite values")
        result[:, 0:3] = rates
    return result


def scale_gate_observations(
    observations: np.ndarray,
    scale: float | np.ndarray,
) -> np.ndarray:
    """Apply a camera-range scale error without adding privileged inputs."""
    result = np.asarray(observations, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != OBSERVATIONS:
        raise ValueError("observations must have shape [agents, 32]")
    scales = np.asarray(scale, dtype=np.float32)
    if scales.ndim == 0:
        scales = np.full(result.shape[0], float(scales), dtype=np.float32)
    if scales.shape != (result.shape[0],):
        raise ValueError("gate observation scales must match the agent axis")
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("gate observation scales must be finite and positive")
    if np.all(scales == 1.0):
        return result
    visible = result[:, 10] > 0.5
    if not np.any(visible):
        return result
    # Public position/rate features are tanh-normalized. A range-scale alias
    # preserves bearing, scales all three vector components and their rates,
    # and inversely scales apparent size.
    for index in (0, 1, 2, 11, 12, 13):
        decoded = np.arctanh(np.clip(result[visible, index], -0.999999, 0.999999))
        result[visible, index] = np.tanh(decoded * scales[visible])
    result[visible, 16] = np.clip(
        result[visible, 16] / scales[visible], 0.0, 1.0)
    return result


def world_frame_gate_features(observations: np.ndarray) -> np.ndarray:
    """Expose world-forward rate and world-vertical gate error legally.

    Camera vectors/rates and the public attitude quaternion are sufficient to
    rotate these features.  Body-right position/rate remain untouched because
    they are the directly useful turn errors.  No native state is consumed.
    """
    result = np.asarray(observations, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != OBSERVATIONS:
        raise ValueError("observations must have shape [agents, 32]")
    for row in result:
        if row[10] <= 0.5:
            continue
        forward_rate, right_rate, down_rate = (
            float(np.arctanh(np.clip(row[0], -0.999999, 0.999999)) * 5.0),
            float(np.arctanh(np.clip(row[1], -0.999999, 0.999999)) * 3.0),
            float(np.arctanh(np.clip(row[2], -0.999999, 0.999999)) * 3.0),
        )
        forward, right, down = (
            float(np.arctanh(np.clip(row[11], -0.999999, 0.999999)) * 10.0),
            float(np.arctanh(np.clip(row[12], -0.999999, 0.999999)) * 5.0),
            float(np.arctanh(np.clip(row[13], -0.999999, 0.999999)) * 5.0),
        )
        qw, qx, qy, qz = (float(value) for value in row[6:10])

        def rotate(vector: tuple[float, float, float]) -> tuple[float, float, float]:
            vx, vy, vz = vector
            tx = 2.0 * (qy * vz - qz * vy)
            ty = 2.0 * (qz * vx - qx * vz)
            tz = 2.0 * (qx * vy - qy * vx)
            return (
                vx + qw * tx + (qy * tz - qz * ty),
                vy + qw * ty + (qz * tx - qx * tz),
                vz + qw * tz + (qx * ty - qy * tx),
            )

        world_rate = rotate((forward_rate, right_rate, -down_rate))
        world_position = rotate((forward, right, -down))
        row[0] = np.tanh(world_rate[0] / 5.0)
        row[2] = np.tanh(world_rate[2] / 3.0)
        row[13] = np.tanh(world_position[2] / 5.0)
    return result


def linear_gate_features(observations: np.ndarray) -> np.ndarray:
    """Decode the six public tanh gate features into clipped linear units.

    Each output remains normalized to the same physical scale as its source
    feature.  This is deterministic observation normalization only: it neither
    combines errors nor produces an action.
    """
    result = np.asarray(observations, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != OBSERVATIONS:
        raise ValueError("observations must have shape [agents, 32]")
    visible = result[:, 10] > 0.5
    if not np.any(visible):
        return result
    for index in (0, 1, 2, 11, 12, 13):
        result[visible, index] = np.clip(
            np.arctanh(np.clip(result[visible, index], -0.999999, 0.999999)),
            -1.0,
            1.0,
        )
    return result


def advance_forward_gate_rate_world_x(
    gate_rate_world_x: np.ndarray,
    observations: np.ndarray,
    *,
    dt_s: float = 1.0 / 60.0,
    hover_thrust: float = 0.27,
    min_thrust: float = 0.18,
    max_thrust: float = 0.42,
    vertical_accel_per_thrust: float = 32.81,
    horizontal_accel_scale: float = 2.60794,
    linear_drag: float = 0.14,
) -> np.ndarray:
    """Propagate observable forward gate rate through the calibrated plant.

    The gate is stationary, so its world-frame relative rate is the negated
    vehicle velocity.  Attitude, prior normalized thrust, and elapsed control
    time are all public inputs.  This deliberately predicts only the forward
    rate; camera-derived right/vertical rates remain untouched.
    """
    values = np.asarray(observations, dtype=np.float32)
    rates = np.asarray(gate_rate_world_x, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != OBSERVATIONS:
        raise ValueError("observations must have shape [agents, 32]")
    if rates.shape != (values.shape[0],):
        raise ValueError("forward gate rates must match the agent axis")
    if not np.isfinite(rates).all() or not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("forward predictor state and dt must be finite")

    thrust_action = np.clip(values[:, LAST_ACTION_START + 2], -1.0, 1.0)
    positive_span = max_thrust - hover_thrust
    negative_span = hover_thrust - min_thrust
    command_thrust = hover_thrust + thrust_action * np.where(
        thrust_action >= 0.0, positive_span, negative_span)
    qw, qx, qy, qz = (values[:, index] for index in range(6, 10))
    thrust_accel = vertical_accel_per_thrust * command_thrust
    # X component of q * (0, 0, thrust_accel) * q^-1.  The calibrated
    # native/live convention negates it so positive pitch brakes world X.
    thrust_world_x = -2.0 * thrust_accel * (qw * qy + qx * qz)
    thrust_world_x *= horizontal_accel_scale
    velocity_world_x = -rates
    accel_world_x = thrust_world_x - linear_drag * velocity_world_x
    return np.asarray(
        -(velocity_world_x + accel_world_x * float(dt_s)), dtype=np.float32)


def _rotate_vectors_by_quaternion(
    vectors: np.ndarray, quaternions: np.ndarray
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    quaternions = np.asarray(quaternions, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape [agents, 3]")
    if quaternions.shape != (vectors.shape[0], 4):
        raise ValueError("quaternions must match the vector agent axis")
    xyz = quaternions[:, 1:4]
    cross = np.cross(xyz, vectors)
    return vectors + 2.0 * (
        quaternions[:, 0:1] * cross + np.cross(xyz, cross))


def initialize_gate_kinematic_state(
    observations: np.ndarray,
    initial_gate_rate_world: np.ndarray,
    initial_gate_position_world_scale: np.ndarray | None = None,
    initial_gate_position_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize a legal stationary-gate state from camera pose and prior rate."""
    values = np.asarray(observations, dtype=np.float32)
    rates = np.asarray(initial_gate_rate_world, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != OBSERVATIONS:
        raise ValueError("observations must have shape [agents, 32]")
    if rates.shape != (values.shape[0], 3):
        raise ValueError("initial gate rates must have shape [agents, 3]")
    if initial_gate_position_world is not None:
        world_position = np.asarray(
            initial_gate_position_world, dtype=np.float32).copy()
        if world_position.shape != (values.shape[0], 3):
            raise ValueError(
                "initial gate world positions must have shape [agents, 3]")
        if not np.isfinite(world_position).all():
            raise ValueError("initial gate world positions must be finite")
    else:
        body_up = np.column_stack((
            np.arctanh(np.clip(values[:, 11], -0.999999, 0.999999)) * 10.0,
            np.arctanh(np.clip(values[:, 12], -0.999999, 0.999999)) * 5.0,
            -np.arctanh(np.clip(values[:, 13], -0.999999, 0.999999)) * 5.0,
        )).astype(np.float32)
        world_position = _rotate_vectors_by_quaternion(body_up, values[:, 6:10])
    if initial_gate_position_world_scale is not None and initial_gate_position_world is None:
        scale = np.asarray(initial_gate_position_world_scale, dtype=np.float32)
        if scale.shape != (values.shape[0], 3):
            raise ValueError(
                "initial gate position scales must have shape [agents, 3]")
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("initial gate position scales must be finite and positive")
        world_position *= scale
    return world_position, rates.copy()


def advance_gate_kinematic_state(
    gate_position_world: np.ndarray,
    gate_rate_world: np.ndarray,
    observations: np.ndarray,
    *,
    dt_s: float = 1.0 / 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate stationary-gate pose/rate from public attitude and prior thrust."""
    position = np.asarray(gate_position_world, dtype=np.float32)
    rate = np.asarray(gate_rate_world, dtype=np.float32)
    values = np.asarray(observations, dtype=np.float32)
    if position.shape != rate.shape or position.shape != (values.shape[0], 3):
        raise ValueError("kinematic state must match the observation agent axis")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("kinematic predictor dt must be finite and positive")

    thrust_action = np.clip(values[:, LAST_ACTION_START + 2], -1.0, 1.0)
    command_thrust = 0.27 + thrust_action * np.where(
        thrust_action >= 0.0, 0.42 - 0.27, 0.27 - 0.18)
    thrust_accel = (32.81 * command_thrust).astype(np.float32)
    thrust_world = _rotate_vectors_by_quaternion(
        np.column_stack((
            np.zeros_like(thrust_accel),
            np.zeros_like(thrust_accel),
            thrust_accel,
        )),
        values[:, 6:10],
    )
    thrust_world[:, 0] *= -2.60794
    thrust_world[:, 1] *= 0.107491
    accel_world = thrust_world
    accel_world[:, 2] -= 8.69
    # Vehicle velocity is the negative stationary-gate rate.
    accel_world += 0.14 * rate
    next_rate = rate - accel_world * float(dt_s)
    next_position = position + 0.5 * (rate + next_rate) * float(dt_s)
    return next_position.astype(np.float32), next_rate.astype(np.float32)


def apply_gate_kinematic_state(
    observations: np.ndarray,
    gate_position_world: np.ndarray,
    gate_rate_world: np.ndarray,
) -> np.ndarray:
    """Encode predicted world gate state into the existing public six slots."""
    result = np.asarray(observations, dtype=np.float32).copy()
    conjugate = result[:, 6:10].copy()
    conjugate[:, 1:4] *= -1.0
    body_position_up = _rotate_vectors_by_quaternion(gate_position_world, conjugate)
    body_rate_up = _rotate_vectors_by_quaternion(gate_rate_world, conjugate)
    result[:, 0] = np.tanh(body_rate_up[:, 0] / 5.0)
    result[:, 1] = np.tanh(body_rate_up[:, 1] / 3.0)
    result[:, 2] = np.tanh(-body_rate_up[:, 2] / 3.0)
    result[:, 11] = np.tanh(body_position_up[:, 0] / 10.0)
    result[:, 12] = np.tanh(body_position_up[:, 1] / 5.0)
    result[:, 13] = np.tanh(-body_position_up[:, 2] / 5.0)
    return result


def reseed_gate_position_world_from_body_bearing(
    gate_position_world: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Correct predicted gate direction from a newly associated public camera pose.

    Monocular range can jump when the detector changes apertures, so retain the
    predictor's world-forward range and adopt only the camera bearing/elevation.
    This is deterministic state estimation; it does not construct an action.
    """
    position = np.asarray(gate_position_world, dtype=np.float32)
    values = np.asarray(observations, dtype=np.float32)
    if position.shape != (values.shape[0], 3) or values.shape[1] != OBSERVATIONS:
        raise ValueError("gate position and observations must share an agent axis")
    result = position.copy()
    visible = values[:, 10] > 0.5
    if not np.any(visible):
        return result

    quaternion = values[visible, 6:10]
    conjugate = quaternion.copy()
    conjugate[:, 1:4] *= -1.0
    predicted_body_up = _rotate_vectors_by_quaternion(
        position[visible], conjugate)
    raw_body_up = np.column_stack((
        np.arctanh(np.clip(values[visible, 11], -0.999999, 0.999999)) * 10.0,
        np.arctanh(np.clip(values[visible, 12], -0.999999, 0.999999)) * 5.0,
        -np.arctanh(np.clip(values[visible, 13], -0.999999, 0.999999)) * 5.0,
    )).astype(np.float32)
    valid = (
        np.isfinite(raw_body_up).all(axis=1)
        & np.isfinite(predicted_body_up).all(axis=1)
        & (raw_body_up[:, 0] > 1e-3)
        & (predicted_body_up[:, 0] > 1e-3)
    )
    if not np.any(valid):
        return result
    corrected_body_up = raw_body_up[valid] * (
        predicted_body_up[valid, 0:1] / raw_body_up[valid, 0:1])
    corrected_world = _rotate_vectors_by_quaternion(
        corrected_body_up, quaternion[valid])
    visible_indices = np.flatnonzero(visible)
    result[visible_indices[valid]] = corrected_world
    return result


def gate_rate_world_from_observation(observation: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (OBSERVATIONS,):
        raise ValueError("observation must contain 32 values")
    forward_rate = float(np.arctanh(np.clip(values[0], -0.999999, 0.999999)) * 5.0)
    right_rate = float(np.arctanh(np.clip(values[1], -0.999999, 0.999999)) * 3.0)
    down_rate = float(np.arctanh(np.clip(values[2], -0.999999, 0.999999)) * 3.0)
    qw, qx, qy, qz = (float(value) for value in values[6:10])
    vx, vy, vz = forward_rate, right_rate, -down_rate
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def scheduled_policy_actions(
    primary: torch.Tensor,
    late: torch.Tensor | None,
    *,
    gate2_elapsed_seconds: float,
    switch_after_seconds: float | None,
    switch_until_seconds: float | None = None,
) -> tuple[torch.Tensor, bool]:
    """Select one complete Puffer output without action-level blending."""
    if late is None or switch_after_seconds is None:
        return primary, False
    if (
        gate2_elapsed_seconds >= switch_after_seconds
        and (
            switch_until_seconds is None
            or gate2_elapsed_seconds < switch_until_seconds
        )
    ):
        return late, True
    return primary, False


def recover_teacher_action(
    policy_action: np.ndarray,
    executed_action: np.ndarray,
    blend: float,
) -> np.ndarray:
    """Recover an offline teacher label from the plant's blended public action."""
    if not 0.0 < blend <= 1.0:
        raise ValueError("teacher recovery blend must be in (0, 1]")
    policy = np.asarray(policy_action, dtype=np.float32)
    executed = np.asarray(executed_action, dtype=np.float32)
    if policy.shape != (ACTIONS,) or executed.shape != (ACTIONS,):
        raise ValueError("policy and executed actions must each contain four values")
    teacher = (executed - (1.0 - blend) * policy) / blend
    return np.clip(teacher, -1.0, 1.0).astype(np.float32)


def assisted_dataset_target(
    policy_action: np.ndarray,
    executed_action: np.ndarray,
    blend: float,
    *,
    target: str,
) -> np.ndarray:
    """Choose a public-action target for an assisted offline rollout."""
    if target == "teacher":
        return recover_teacher_action(policy_action, executed_action, blend)
    if target == "executed":
        action = np.asarray(executed_action, dtype=np.float32)
        if action.shape != (ACTIONS,) or not np.isfinite(action).all():
            raise ValueError("executed action must contain four finite values")
        return np.clip(action, -1.0, 1.0).astype(np.float32)
    raise ValueError(f"unknown assisted dataset target: {target}")


def _buffer(pointer: int, count: int) -> np.ndarray:
    return np.ctypeslib.as_array((ctypes.c_float * count).from_address(pointer))


def _flat_vec_log(pufferl_module, log: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in pufferl_module.unroll_nested_dict(log):
        normalized = str(key)
        if "/" not in normalized:
            normalized = f"env/{normalized}"
        metrics[normalized] = float(value)
    return metrics


def _overrides(args: argparse.Namespace) -> list[str]:
    start_elapsed_time = float(getattr(args, "start_elapsed_time", 3.25))
    time_limit_seconds = float(getattr(args, "time_limit_seconds", 20.0))
    gate1_x = float(getattr(args, "gate1_x", 14.74))
    gate1_y = float(getattr(args, "gate1_y", 8.70))
    gate1_z = float(getattr(args, "gate1_z", 1.095))
    crash_height = float(getattr(args, "crash_height", -10.0))
    safety_altitude = float(getattr(args, "safety_altitude", -8.0))
    plant_lateral_accel_scale = float(getattr(
        args, "plant_lateral_accel_scale", 0.107491))
    values = [
        "--seed", str(args.seed),
        "--vec.total-agents", str(args.episodes),
        "--vec.num-buffers", "4",
        "--vec.num-threads", "8",
        "--env.num-gates", "2",
        "--env.use-custom-start", "1",
        "--env.start-gate-index", "1",
        "--env.start-elapsed-time", str(start_elapsed_time),
        "--env.mixed-start-curriculum", "0",
        "--env.start-x", "0", "--env.start-y", "0", "--env.start-z", "0",
        "--env.reset-position-noise-xy", "0",
        "--env.reset-position-noise-z", "0",
        "--env.start-vx", "4.677", "--env.start-vy", "0.006", "--env.start-vz", "0.173",
        "--env.start-qw", "0.998520", "--env.start-qx", "0.052695",
        "--env.start-qy", "-0.013450", "--env.start-qz", "0.000931",
        "--env.start-wx", "0.1226", "--env.start-wy", "-0.0006", "--env.start-wz", "0",
        "--env.use-custom-gate-layout", "1",
        "--env.gate0-x", "0", "--env.gate0-y", "0", "--env.gate0-z", "0",
        "--env.gate1-x", str(gate1_x),
        "--env.gate1-y", str(gate1_y),
        "--env.gate1-z", str(gate1_z),
        "--env.gate-radius", str(args.gate_radius),
        "--env.gate0-radius", str(args.gate_radius),
        "--env.gate1-radius", str(args.gate_radius),
        "--env.observable-gate-progress", "1",
        "--env.observable-gate-index-denominator", "6",
        "--env.observable-gate-phase-onehot", "1",
        "--env.sitl-gate-obs-sample-interval-steps", "4",
        "--env.sitl-gate-obs-dropout-range-m", "4.25",
        "--env.sitl-gate-obs-dropout-from-index", "1",
        "--env.sitl-gate-motion-predict-dropout", str(int(getattr(
            args, "predict_gate_motion_dropout", False))),
        "--env.sitl-gate-motion-control-accel-gain", str(getattr(
            args, "gate_motion_control_accel_gain", 0.0)),
        "--env.sitl-gate-motion-initial-vx", str(getattr(
            args, "gate_motion_initial_vx", 0.0)),
        "--env.sitl-gate-motion-initial-vy", str(getattr(
            args, "gate_motion_initial_vy", 0.0)),
        "--env.sitl-gate-motion-initial-vz", str(getattr(
            args, "gate_motion_initial_vz", 0.0)),
        "--env.sitl-lateral-accel-scale", str(plant_lateral_accel_scale),
        "--env.max-steps", "1200",
        "--env.time-limit-seconds", str(time_limit_seconds),
        "--env.crash-height", str(crash_height),
        "--env.safety-altitude", str(safety_altitude),
        "--env.pos-bound", "45",
        "--env.teacher-action-blend", str(args.training_teacher_blend),
        "--env.teacher-action-blend-from-step", str(int(round(
            float(getattr(
                args, "training_teacher_blend_after_gate2_seconds", 0.0
            )) * 60.0
        ))),
    ]
    if args.training_teacher_blend > 0.0:
        values.extend([
            "--env.teacher-pitch-from-gate-index", "1",
            "--env.teacher-pitch-speed-control", "1",
            "--env.teacher-pitch-speed-target-m-s", str(getattr(
                args, "teacher_pitch_speed_target_m_s", 1.30)),
            "--env.teacher-pitch-speed-gain", "0.35",
            "--env.teacher-pitch-speed-scale", "0.24",
            "--env.teacher-roll-from-gate-index", "1",
            "--env.teacher-roll-until-gate-index", "2",
            "--env.teacher-thrust-from-gate-index", "1",
            "--env.teacher-yaw-control", "1",
            "--env.teacher-yaw-from-gate-index", "1",
            "--env.teacher-yaw-action", "0",
            "--env.teacher-roll-per-m", str(getattr(
                args, "teacher_roll_per_m", 0.35)),
            "--env.teacher-roll-rate-per-m-s", str(getattr(
                args, "teacher_roll_rate_per_m_s", 0.10)),
            "--env.teacher-thrust-bias", str(getattr(
                args, "teacher_thrust_bias", 0.0)),
            "--env.teacher-thrust-per-m", str(getattr(
                args, "teacher_thrust_per_m", 0.30)),
            "--env.teacher-thrust-rate-per-m-s", str(getattr(
                args, "teacher_thrust_rate_per_m_s", 0.10)),
            "--env.teacher-thrust-world-frame", "1",
        ])
    if args.perturbed:
        scale = float(args.perturbation_scale)
        components = set(getattr(
            args, "perturbation_components", ("start", "gate", "plant")))
        start_dimensions = set(getattr(
            args, "start_perturbation_dimensions",
            START_PERTURBATION_DIMENSIONS))
        start_scale = scale if "start" in components else 0.0
        gate_scale = scale if "gate" in components else 0.0
        plant_scale = scale if "plant" in components else 0.0
        elapsed_scale = start_scale if "elapsed" in start_dimensions else 0.0
        position_scale = start_scale if "position" in start_dimensions else 0.0
        velocity_scale = start_scale if "velocity" in start_dimensions else 0.0
        attitude_scale = start_scale if "attitude" in start_dimensions else 0.0
        rates_scale = start_scale if "rates" in start_dimensions else 0.0
        values.extend([
            "--env.start-elapsed-time-jitter", str(0.15 * elapsed_scale),
            "--env.start-x-jitter", str(0.10 * position_scale),
            "--env.start-y-jitter", str(0.10 * position_scale),
            "--env.start-z-jitter", str(0.10 * position_scale),
            "--env.start-vx-jitter", str(0.20 * velocity_scale),
            "--env.start-vy-jitter", str(0.10 * velocity_scale),
            "--env.start-vz-jitter", str(0.10 * velocity_scale),
            "--env.start-roll-jitter-rad", str(0.015 * attitude_scale),
            "--env.start-pitch-jitter-rad", str(0.015 * attitude_scale),
            "--env.start-yaw-jitter-rad", str(0.015 * attitude_scale),
            "--env.start-wx-jitter", str(0.03 * rates_scale),
            "--env.start-wy-jitter", str(0.03 * rates_scale),
            "--env.start-wz-jitter", str(0.03 * rates_scale),
            "--env.gate-position-domain-randomize", str(int(gate_scale > 0.0)),
            "--env.gate-position-domain-randomize-probability", "1",
            "--env.gate-position-randomize-from-index", "1",
            "--env.gate-position-jitter-x", str(0.30 * gate_scale),
            "--env.gate-position-jitter-y", str(0.40 * gate_scale),
            "--env.gate-position-jitter-z", str(0.30 * gate_scale),
            "--env.sitl-plant-domain-randomize", str(int(plant_scale > 0.0)),
            "--env.sitl-rate-gain-jitter-frac", str(0.05 * plant_scale),
            "--env.sitl-hover-thrust-jitter", str(0.01 * plant_scale),
            "--env.sitl-rate-lag-jitter-frac", str(0.05 * plant_scale),
            "--env.sitl-linear-drag-jitter-frac", str(0.05 * plant_scale),
        ])
    else:
        values.extend([
            "--env.start-elapsed-time-jitter", "0",
            "--env.start-x-jitter", "0", "--env.start-y-jitter", "0",
            "--env.start-z-jitter", "0", "--env.start-vx-jitter", "0",
            "--env.start-vy-jitter", "0", "--env.start-vz-jitter", "0",
            "--env.start-roll-jitter-rad", "0",
            "--env.start-pitch-jitter-rad", "0",
            "--env.start-yaw-jitter-rad", "0",
            "--env.start-wx-jitter", "0", "--env.start-wy-jitter", "0",
            "--env.start-wz-jitter", "0",
            "--env.gate-position-domain-randomize", "0",
            "--env.sitl-plant-domain-randomize", "0",
        ])
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("official_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3317)
    parser.add_argument("--gate-radius", type=float, default=0.75)
    parser.add_argument("--gate1-x", type=float, default=14.74)
    parser.add_argument("--gate1-y", type=float, default=8.70)
    parser.add_argument("--gate1-z", type=float, default=1.095)
    parser.add_argument(
        "--start-elapsed-time", type=float, default=3.25,
        help="Native Gate-2 start time; keep consistent with the prefix clock.",
    )
    parser.add_argument(
        "--time-limit-seconds", type=float, default=20.0,
        help="Denominator for the native elapsed-time observation.",
    )
    parser.add_argument(
        "--crash-height", type=float, default=-10.0,
        help=(
            "Offline native lower-bound collision plane. Use a tighter value to "
            "screen the official arena's measured altitude envelope."
        ),
    )
    parser.add_argument(
        "--safety-altitude", type=float, default=-8.0,
        help=(
            "Offline native altitude at which the low-floor safety shaping begins."
        ),
    )
    parser.add_argument(
        "--plant-lateral-accel-scale", type=float, default=0.107491,
        help=(
            "Offline native lateral plant scale. The legal observation predictor "
            "retains its calibrated scale, allowing explicit transfer-mismatch tests."
        ),
    )
    parser.add_argument(
        "--truncate-prefix-at-gate-advance", action="store_true",
        help=(
            "Use only the leading official-index-0 samples from a report that "
            "continued beyond Gate 1."
        ),
    )
    parser.add_argument(
        "--policy-gate-observation-scale", type=float, default=1.0,
        help=(
            "Command-free sensitivity screen: scale legal gate vectors/rates "
            "before Puffer inference while leaving the native plant unchanged."
        ),
    )
    parser.add_argument("--policy-gate-observation-scale-min", type=float)
    parser.add_argument("--policy-gate-observation-scale-max", type=float)
    parser.add_argument("--policy-world-frame-gate-features", action="store_true")
    parser.add_argument("--policy-linear-gate-features", action="store_true")
    parser.add_argument("--policy-predict-forward-gate-rate", action="store_true")
    parser.add_argument("--policy-predict-gate-kinematics", action="store_true")
    parser.add_argument(
        "--policy-bearing-reseed-after-gate2-seconds", type=float,
        help=(
            "Offline proxy for the first trustworthy newly associated gate: "
            "preserve predicted forward range and reseed right/down bearing from "
            "the public camera once at this Gate-2 elapsed time."
        ),
    )
    parser.add_argument(
        "--policy-initial-gate-position-world-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help=(
            "Calibrate the initial public camera-derived world gate vector before "
            "the legal stationary-gate predictor starts."
        ),
    )
    parser.add_argument(
        "--policy-initial-gate-position-world",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Use a legally calibrated initial relative Gate-2 world vector instead "
            "of the potentially stale first post-transition camera association."
        ),
    )
    parser.add_argument("--carry-prefix-gate-rates", action="store_true")
    parser.add_argument("--perturbed", action="store_true")
    parser.add_argument(
        "--perturbation-scale", type=float, default=1.0,
        help="Scale every explicit start, gate, and plant perturbation.",
    )
    parser.add_argument(
        "--perturbation-components", nargs="+",
        choices=("start", "gate", "plant"),
        default=("start", "gate", "plant"),
        help="Select perturbation families; the default enables all three.",
    )
    parser.add_argument(
        "--start-perturbation-dimensions", nargs="+",
        choices=START_PERTURBATION_DIMENSIONS,
        default=START_PERTURBATION_DIMENSIONS,
        help=(
            "Select start-state jitter groups when the start perturbation family is "
            "enabled; the default enables all five groups."
        ),
    )
    parser.add_argument(
        "--training-teacher-blend", type=float, default=0.0,
        help=(
            "Offline DAgger capture only. Admission screens must leave this at zero; "
            "a positive value requires --dataset-output."
        ),
    )
    parser.add_argument(
        "--training-teacher-blend-after-gate2-seconds", type=float, default=0.0,
        help=(
            "Training-only delay before native teacher intervention; deployment "
            "and admission still require blend zero."
        ),
    )
    parser.add_argument("--teacher-pitch-speed-target-m-s", type=float, default=1.30)
    parser.add_argument("--teacher-roll-per-m", type=float, default=0.35)
    parser.add_argument("--teacher-roll-rate-per-m-s", type=float, default=0.10)
    parser.add_argument("--teacher-thrust-bias", type=float, default=0.0)
    parser.add_argument("--teacher-thrust-per-m", type=float, default=0.30)
    parser.add_argument("--teacher-thrust-rate-per-m-s", type=float, default=0.10)
    parser.add_argument(
        "--dataset-output", type=Path,
        help="Write prefixed teacher-label records recovered during an assisted rollout.",
    )
    parser.add_argument(
        "--dataset-target", choices=("teacher", "executed"), default="teacher",
        help=(
            "Store either the recovered full teacher action or the public action "
            "that actually drove the assisted offline plant."
        ),
    )
    parser.add_argument(
        "--trace-output", type=Path,
        help=(
            "Write a command-free NPZ trace of active public observations, complete "
            "Puffer actions, and terminal post-observations for offline diagnosis."
        ),
    )
    parser.add_argument(
        "--reset-recurrent-at-gate2", action="store_true",
        help="Start the secondary Puffer recurrent state at zero on Gate-2 activation.",
    )
    parser.add_argument("--predict-gate-motion-dropout", action="store_true")
    parser.add_argument("--gate-motion-control-accel-gain", type=float, default=0.0)
    parser.add_argument(
        "--late-checkpoint", type=Path,
        help=(
            "Optional second recurrent Puffer checkpoint. Both policies are warmed and "
            "advanced on every public observation; this checkpoint's complete action "
            "vector is selected after --switch-after-gate2-seconds."
        ),
    )
    parser.add_argument(
        "--switch-after-gate2-seconds", type=float,
        help="Teacher-free elapsed-time selector for --late-checkpoint.",
    )
    parser.add_argument(
        "--switch-until-gate2-seconds", type=float,
        help=(
            "Optional end of the late-policy recovery window; afterward the primary "
            "Puffer checkpoint is selected again."
        ),
    )
    parser.add_argument(
        "--activation-delay-checkpoint", type=Path,
        help=(
            "Whole-action Puffer checkpoint selected until the Gate-2 activation "
            "delay elapses; both recurrent policies still consume every legal tick."
        ),
    )
    parser.add_argument(
        "--gate2-activation-delay-seconds", type=float, default=0.0,
        help="Offline proxy for delayed public association of the newly active gate.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not np.isfinite(args.start_elapsed_time) or args.start_elapsed_time < 0.0:
        parser.error("--start-elapsed-time must be finite and nonnegative")
    if not np.isfinite(args.time_limit_seconds) or args.time_limit_seconds <= 0.0:
        parser.error("--time-limit-seconds must be finite and positive")
    if args.start_elapsed_time >= args.time_limit_seconds:
        parser.error("--start-elapsed-time must be less than --time-limit-seconds")
    if (
        not np.isfinite(args.crash_height)
        or not np.isfinite(args.safety_altitude)
        or args.crash_height >= 0.0
        or args.safety_altitude <= args.crash_height
        or args.safety_altitude >= 0.0
    ):
        parser.error(
            "altitude envelope requires crash-height < safety-altitude < 0")
    if (
        not np.isfinite(args.plant_lateral_accel_scale)
        or args.plant_lateral_accel_scale <= 0.0
    ):
        parser.error("--plant-lateral-accel-scale must be finite and positive")
    if not np.isfinite((args.gate1_x, args.gate1_y, args.gate1_z)).all():
        parser.error("Gate-2 coordinates must be finite")
    if (
        not np.isfinite(args.policy_initial_gate_position_world_scale).all()
        or any(value <= 0.0 for value in args.policy_initial_gate_position_world_scale)
    ):
        parser.error("initial gate position world scales must be finite and positive")
    if (
        args.policy_initial_gate_position_world is not None
        and not np.isfinite(args.policy_initial_gate_position_world).all()
    ):
        parser.error("initial gate world position must be finite")
    if (
        args.policy_bearing_reseed_after_gate2_seconds is not None
        and (
            not np.isfinite(args.policy_bearing_reseed_after_gate2_seconds)
            or args.policy_bearing_reseed_after_gate2_seconds < 0.0
        )
    ):
        parser.error("bearing reseed time must be finite and nonnegative")
    if (
        args.policy_bearing_reseed_after_gate2_seconds is not None
        and not args.policy_predict_gate_kinematics
    ):
        parser.error("bearing reseed requires the full gate-kinematic predictor")
    if (
        not np.isfinite(args.policy_gate_observation_scale)
        or args.policy_gate_observation_scale <= 0.0
    ):
        parser.error("--policy-gate-observation-scale must be finite and positive")
    if (args.policy_gate_observation_scale_min is None) != (
        args.policy_gate_observation_scale_max is None
    ):
        parser.error("gate observation scale min/max must be supplied together")
    if args.policy_gate_observation_scale_min is not None:
        if (
            not np.isfinite(args.policy_gate_observation_scale_min)
            or not np.isfinite(args.policy_gate_observation_scale_max)
            or args.policy_gate_observation_scale_min <= 0.0
            or args.policy_gate_observation_scale_max
            < args.policy_gate_observation_scale_min
        ):
            parser.error("invalid gate observation scale range")
    if not np.isfinite(args.perturbation_scale) or args.perturbation_scale <= 0.0:
        parser.error("--perturbation-scale must be finite and positive")
    if not np.isfinite(args.gate_motion_control_accel_gain):
        parser.error("--gate-motion-control-accel-gain must be finite")
    if not 0.0 <= args.training_teacher_blend <= 1.0:
        parser.error("--training-teacher-blend must be in [0, 1]")
    if (
        not np.isfinite(args.training_teacher_blend_after_gate2_seconds)
        or args.training_teacher_blend_after_gate2_seconds < 0.0
    ):
        parser.error("training teacher delay must be finite and nonnegative")
    if (
        not np.isfinite(args.teacher_pitch_speed_target_m_s)
        or args.teacher_pitch_speed_target_m_s <= 0.0
        or not np.isfinite(args.teacher_roll_per_m)
        or not np.isfinite(args.teacher_roll_rate_per_m_s)
        or not np.isfinite(args.teacher_thrust_bias)
        or not np.isfinite(args.teacher_thrust_per_m)
        or not np.isfinite(args.teacher_thrust_rate_per_m_s)
    ):
        parser.error("teacher speed and roll gains must be finite and speed positive")
    if args.training_teacher_blend > 0.0 and args.dataset_output is None:
        parser.error("positive training teacher blend requires --dataset-output")
    if args.dataset_output is not None and args.training_teacher_blend <= 0.0:
        parser.error("--dataset-output requires a positive training teacher blend")
    if (args.late_checkpoint is None) != (args.switch_after_gate2_seconds is None):
        parser.error(
            "--late-checkpoint and --switch-after-gate2-seconds must be supplied together"
        )
    if args.switch_after_gate2_seconds is not None and args.switch_after_gate2_seconds < 0.0:
        parser.error("--switch-after-gate2-seconds must be nonnegative")
    if args.switch_until_gate2_seconds is not None:
        if args.switch_after_gate2_seconds is None:
            parser.error("--switch-until-gate2-seconds requires a scheduled late policy")
        if args.switch_until_gate2_seconds <= args.switch_after_gate2_seconds:
            parser.error(
                "--switch-until-gate2-seconds must exceed --switch-after-gate2-seconds"
            )
    if (
        not np.isfinite(args.gate2_activation_delay_seconds)
        or args.gate2_activation_delay_seconds < 0.0
    ):
        parser.error("--gate2-activation-delay-seconds must be finite and nonnegative")
    if (
        args.gate2_activation_delay_seconds > 0.0
        and args.activation_delay_checkpoint is None
    ):
        parser.error("a positive Gate-2 activation delay requires its Puffer checkpoint")

    from pufferlib import _C, pufferl

    prefix_observations, prefix_actions = load_policy_prefix(
        args.official_report,
        stop_at_gate_advance=args.truncate_prefix_at_gate_advance,
    )
    if args.carry_prefix_gate_rates:
        (
            args.gate_motion_initial_vx,
            args.gate_motion_initial_vy,
            args.gate_motion_initial_vz,
        ) = gate_rate_world_from_observation(prefix_observations[-1])
    checkpoint = CheckpointPolicy.load(
        str(args.checkpoint), input_dim=OBSERVATIONS, num_layers=3,
        layout_precision_bytes=4)
    device = torch.device(args.device)
    model = SequencePufferNet(checkpoint).to(device).eval()
    with torch.no_grad():
        prefix_tensor = torch.from_numpy(prefix_observations).to(device).unsqueeze(0)
        state = model.initial_state(1, device)
        replayed, state = model.forward_chunk(prefix_tensor, state)
        replayed_prefix = torch.clamp(replayed[0], -1.0, 1.0).cpu().numpy()
    prefix_max_action_error = float(np.max(np.abs(replayed_prefix - prefix_actions)))
    if args.reset_recurrent_at_gate2:
        state = model.initial_state(1, device)
    late_model = None
    late_state = None
    late_prefix_max_action_error = None
    if args.late_checkpoint is not None:
        late_checkpoint = CheckpointPolicy.load(
            str(args.late_checkpoint), input_dim=OBSERVATIONS, num_layers=3,
            layout_precision_bytes=4)
        late_model = SequencePufferNet(late_checkpoint).to(device).eval()
        with torch.no_grad():
            late_prefix_tensor = torch.from_numpy(prefix_observations).to(device).unsqueeze(0)
            late_state = late_model.initial_state(1, device)
            late_replayed, late_state = late_model.forward_chunk(
                late_prefix_tensor, late_state)
            late_replayed_prefix = torch.clamp(
                late_replayed[0], -1.0, 1.0).cpu().numpy()
        late_prefix_max_action_error = float(
            np.max(np.abs(late_replayed_prefix - prefix_actions)))
        if args.reset_recurrent_at_gate2:
            late_state = late_model.initial_state(1, device)

    activation_delay_model = None
    activation_delay_state = None
    activation_delay_prefix_max_action_error = None
    if args.activation_delay_checkpoint is not None:
        activation_delay_checkpoint = CheckpointPolicy.load(
            str(args.activation_delay_checkpoint), input_dim=OBSERVATIONS,
            num_layers=3, layout_precision_bytes=4)
        activation_delay_model = SequencePufferNet(
            activation_delay_checkpoint).to(device).eval()
        with torch.no_grad():
            activation_prefix_tensor = torch.from_numpy(
                prefix_observations).to(device).unsqueeze(0)
            activation_delay_state = activation_delay_model.initial_state(1, device)
            activation_replayed, activation_delay_state = (
                activation_delay_model.forward_chunk(
                    activation_prefix_tensor, activation_delay_state)
            )
            activation_replayed_prefix = torch.clamp(
                activation_replayed[0], -1.0, 1.0).cpu().numpy()
        activation_delay_prefix_max_action_error = float(np.max(np.abs(
            activation_replayed_prefix - prefix_actions)))

    overrides = _overrides(args)
    cfg = _load_config(pufferl, "drone_race_full_policy_six_gate_bootstrap", overrides)
    cfg.setdefault("env", {})["evaluation_episode_limit"] = 1
    cfg["env"]["evaluation_episode_offset"] = 0
    if getattr(_C, "precision_bytes", 0) != 4:
        raise RuntimeError("prefixed evaluation requires the FP32 native backend")
    vec = _C.create_vec(cfg, gpu=0)
    vec.reset()
    if vec.obs_size != OBSERVATIONS or vec.num_atns != ACTIONS:
        vec.close()
        raise RuntimeError("native environment does not expose the 32x4 policy ABI")
    observations = _buffer(
        vec.obs_ptr, vec.total_agents * vec.obs_size).reshape(vec.total_agents, vec.obs_size)
    terminals = _buffer(vec.terminals_ptr, vec.total_agents)
    if args.policy_gate_observation_scale_min is None:
        policy_gate_observation_scales = np.full(
            vec.total_agents,
            args.policy_gate_observation_scale,
            dtype=np.float32,
        )
    else:
        policy_gate_observation_scales = np.random.default_rng(
            args.seed ^ 0x56413252
        ).uniform(
            args.policy_gate_observation_scale_min,
            args.policy_gate_observation_scale_max,
            size=vec.total_agents,
        ).astype(np.float32)
    state = state.repeat(1, vec.total_agents, 1)
    if late_state is not None:
        late_state = late_state.repeat(1, vec.total_agents, 1)
    if activation_delay_state is not None:
        activation_delay_state = activation_delay_state.repeat(
            1, vec.total_agents, 1)
    completed = np.zeros(vec.total_agents, dtype=np.bool_)
    predicted_forward_gate_rate = np.full(
        vec.total_agents,
        float(args.gate_motion_initial_vx),
        dtype=np.float32,
    )
    predicted_gate_position: np.ndarray | None = None
    predicted_gate_rate = np.tile(
        np.asarray([
            args.gate_motion_initial_vx,
            args.gate_motion_initial_vy,
            args.gate_motion_initial_vz,
        ], dtype=np.float32),
        (vec.total_agents, 1),
    )
    initial_gate_position_world_scale = np.tile(
        np.asarray(
            args.policy_initial_gate_position_world_scale, dtype=np.float32),
        (vec.total_agents, 1),
    )
    initial_gate_position_world = (
        None
        if args.policy_initial_gate_position_world is None
        else np.tile(
            np.asarray(args.policy_initial_gate_position_world, dtype=np.float32),
            (vec.total_agents, 1),
        )
    )
    bearing_reseeded = False
    dagger_records: list[list[np.ndarray]] = [list() for _ in range(vec.total_agents)]
    trace_observations: list[np.ndarray] = []
    trace_actions: list[np.ndarray] = []
    trace_post_observations: list[np.ndarray] = []
    trace_agent_indices: list[np.ndarray] = []
    trace_steps: list[np.ndarray] = []
    trace_terminals: list[np.ndarray] = []
    first_observation = gate2_first_observation(
        observations,
        prefix_actions[-1],
        (
            prefix_observations[-1, 0:3]
            if args.carry_prefix_gate_rates else None
        ),
    )
    actions = torch.zeros((vec.total_agents, ACTIONS), dtype=torch.float32)
    started = time.time()
    steps = 0
    late_selected_steps = 0
    activation_delay_selected_steps = 0
    try:
        for steps in range(1, int(cfg["env"]["max_steps"]) + 3):
            raw_current = first_observation if steps == 1 else observations.copy()
            current = raw_current.copy()
            current = scale_gate_observations(
                current, policy_gate_observation_scales)
            if args.policy_predict_gate_kinematics:
                if predicted_gate_position is None:
                    predicted_gate_position, predicted_gate_rate = (
                        initialize_gate_kinematic_state(
                            current,
                            predicted_gate_rate,
                            initial_gate_position_world_scale,
                            initial_gate_position_world,
                        )
                    )
                elif steps > 1:
                    predicted_gate_position, predicted_gate_rate = (
                        advance_gate_kinematic_state(
                            predicted_gate_position,
                            predicted_gate_rate,
                            current,
                            dt_s=float(cfg["env"]["dt"]),
                        )
                        )
                if (
                    not bearing_reseeded
                    and args.policy_bearing_reseed_after_gate2_seconds is not None
                    and (steps - 1) * float(cfg["env"]["dt"])
                    >= args.policy_bearing_reseed_after_gate2_seconds
                ):
                    predicted_gate_position = (
                        reseed_gate_position_world_from_body_bearing(
                            predicted_gate_position, raw_current)
                    )
                    bearing_reseeded = True
                current = apply_gate_kinematic_state(
                    current, predicted_gate_position, predicted_gate_rate)
            if args.policy_predict_forward_gate_rate and steps > 1:
                predicted_forward_gate_rate = advance_forward_gate_rate_world_x(
                    predicted_forward_gate_rate,
                    current,
                    dt_s=float(cfg["env"]["dt"]),
                )
            if args.policy_world_frame_gate_features:
                current = world_frame_gate_features(current)
            if args.policy_predict_forward_gate_rate:
                if not args.policy_world_frame_gate_features:
                    raise RuntimeError(
                        "forward gate-rate prediction requires world-frame features")
                current[:, 0] = np.tanh(predicted_forward_gate_rate / 5.0)
            if args.policy_linear_gate_features:
                current = linear_gate_features(current)
            current_tensor = torch.from_numpy(current).to(device).unsqueeze(1)
            with torch.no_grad():
                action_values, state = model.forward_chunk(current_tensor, state)
                action_values = torch.clamp(action_values[:, 0], -1.0, 1.0)
                late_action_values = None
                if late_model is not None:
                    assert late_state is not None
                    late_outputs, late_state = late_model.forward_chunk(
                        current_tensor, late_state)
                    late_action_values = torch.clamp(
                        late_outputs[:, 0], -1.0, 1.0)
                activation_delay_action_values = None
                if activation_delay_model is not None:
                    assert activation_delay_state is not None
                    activation_delay_tensor = torch.from_numpy(
                        raw_current).to(device).unsqueeze(1)
                    activation_outputs, activation_delay_state = (
                        activation_delay_model.forward_chunk(
                            activation_delay_tensor, activation_delay_state)
                    )
                    activation_delay_action_values = torch.clamp(
                        activation_outputs[:, 0], -1.0, 1.0)
                action_values, late_selected = scheduled_policy_actions(
                    action_values,
                    late_action_values,
                    gate2_elapsed_seconds=(steps - 1) * float(cfg["env"]["dt"]),
                    switch_after_seconds=args.switch_after_gate2_seconds,
                    switch_until_seconds=args.switch_until_gate2_seconds,
                )
                if late_selected:
                    late_selected_steps += 1
                if (
                    activation_delay_action_values is not None
                    and (steps - 1) * float(cfg["env"]["dt"])
                    < args.gate2_activation_delay_seconds
                ):
                    action_values = activation_delay_action_values
                    activation_delay_selected_steps += 1
            actions.copy_(action_values.cpu())
            policy_actions = actions.numpy().copy()
            active_before_step = ~completed
            vec.cpu_step(actions.data_ptr())
            terminal_now = terminals >= 0.5
            if args.trace_output is not None:
                active_indices = np.flatnonzero(active_before_step)
                trace_observations.append(current[active_indices].copy())
                trace_actions.append(policy_actions[active_indices].copy())
                trace_post_observations.append(observations[active_indices].copy())
                trace_agent_indices.append(active_indices.astype(np.int32))
                trace_steps.append(np.full(
                    len(active_indices), steps, dtype=np.int32))
                trace_terminals.append(
                    terminal_now[active_indices].astype(np.bool_))
            if args.dataset_output is not None:
                label_available = active_before_step & ~terminal_now
                for agent_index in np.flatnonzero(label_available):
                    target_action = assisted_dataset_target(
                        policy_actions[agent_index],
                        observations[agent_index, LAST_ACTION_START:LAST_ACTION_START + ACTIONS],
                        args.training_teacher_blend,
                        target=args.dataset_target,
                    )
                    record = np.zeros(OBSERVATIONS + ACTIONS + 1, dtype=np.float32)
                    record[:OBSERVATIONS] = current[agent_index]
                    record[OBSERVATIONS:OBSERVATIONS + ACTIONS] = target_action
                    dagger_records[agent_index].append(record)
            completed |= terminal_now
            if bool(np.all(completed)):
                break
        metrics = _flat_vec_log(pufferl, dict(vec.log()))
    finally:
        vec.close()
    _add_derived_metrics(metrics)
    if metrics.get("env/n", 0.0) != float(args.episodes):
        raise RuntimeError(
            f"expected {args.episodes} completed episodes, got {metrics.get('env/n', 0.0)}")

    dataset_metadata = None
    if args.dataset_output is not None:
        prefix = np.zeros(
            (len(prefix_observations), OBSERVATIONS + ACTIONS + 1), dtype=np.float32)
        prefix[:, :OBSERVATIONS] = prefix_observations
        prefix[:, OBSERVATIONS:OBSERVATIONS + ACTIONS] = prefix_actions
        prefix[0, -1] = 1.0
        episodes: list[np.ndarray] = []
        lengths: list[int] = []
        for agent_index, rows in enumerate(dagger_records):
            if not rows:
                raise RuntimeError(f"agent {agent_index} produced no DAgger labels")
            episode = np.concatenate((prefix, np.stack(rows)), axis=0)
            episodes.append(episode)
            lengths.append(len(episode))
        dataset = np.concatenate(episodes, axis=0)
        args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
        dataset.tofile(args.dataset_output)
        dataset_metadata = {
            "path": str(args.dataset_output),
            "sha256": sha256_file(args.dataset_output),
            "records": int(len(dataset)),
            "episodes": len(episodes),
            "minimum_episode_length": min(lengths),
            "maximum_episode_length": max(lengths),
            "target_source": (
                "recovered_training_teacher_from_public_executed_action"
                if args.dataset_target == "teacher"
                else "public_executed_assisted_action"
            ),
            "privileged_student_inputs": False,
        }

    trace_metadata = None
    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace_output,
            observations=np.concatenate(trace_observations),
            actions=np.concatenate(trace_actions),
            post_observations=np.concatenate(trace_post_observations),
            agent_indices=np.concatenate(trace_agent_indices),
            steps=np.concatenate(trace_steps),
            terminals=np.concatenate(trace_terminals),
        )
        trace_metadata = {
            "path": str(args.trace_output),
            "sha256": sha256_file(args.trace_output),
            "records": int(sum(len(value) for value in trace_agent_indices)),
            "public_observations_only": True,
            "puffer_actions_only": True,
        }

    report = {
        "metadata": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "official_report": str(args.official_report),
            "official_report_sha256": sha256_file(args.official_report),
            "action_mode": "deterministic_mean",
            "teacher_action_blend": args.training_teacher_blend,
            "teacher_action_blend_after_gate2_seconds": (
                args.training_teacher_blend_after_gate2_seconds),
            "teacher_pitch_speed_target_m_s": args.teacher_pitch_speed_target_m_s,
            "teacher_roll_per_m": args.teacher_roll_per_m,
            "teacher_roll_rate_per_m_s": args.teacher_roll_rate_per_m_s,
            "teacher_thrust_bias": args.teacher_thrust_bias,
            "teacher_thrust_per_m": args.teacher_thrust_per_m,
            "teacher_thrust_rate_per_m_s": args.teacher_thrust_rate_per_m_s,
            "admission_policy_only": args.training_teacher_blend == 0.0,
            "prefixed_recurrent_state": not args.reset_recurrent_at_gate2,
            "reset_recurrent_at_gate2": args.reset_recurrent_at_gate2,
            "predict_gate_motion_dropout": args.predict_gate_motion_dropout,
            "gate_motion_control_accel_gain": args.gate_motion_control_accel_gain,
            "prefix_samples": int(len(prefix_observations)),
            "truncate_prefix_at_gate_advance": args.truncate_prefix_at_gate_advance,
            "start_elapsed_time": args.start_elapsed_time,
            "time_limit_seconds": args.time_limit_seconds,
            "crash_height": args.crash_height,
            "safety_altitude": args.safety_altitude,
            "plant_lateral_accel_scale": args.plant_lateral_accel_scale,
            "gate1_xyz": [args.gate1_x, args.gate1_y, args.gate1_z],
            "policy_gate_observation_scale": args.policy_gate_observation_scale,
            "policy_gate_observation_scale_min": (
                args.policy_gate_observation_scale_min),
            "policy_gate_observation_scale_max": (
                args.policy_gate_observation_scale_max),
            "policy_world_frame_gate_features": (
                args.policy_world_frame_gate_features),
            "policy_linear_gate_features": args.policy_linear_gate_features,
            "policy_predict_forward_gate_rate": (
                args.policy_predict_forward_gate_rate),
            "policy_predict_gate_kinematics": (
                args.policy_predict_gate_kinematics),
            "policy_bearing_reseed_after_gate2_seconds": (
                args.policy_bearing_reseed_after_gate2_seconds),
            "policy_bearing_reseeded": bearing_reseeded,
            "policy_initial_gate_position_world_scale": list(
                args.policy_initial_gate_position_world_scale),
            "policy_initial_gate_position_world": (
                None
                if args.policy_initial_gate_position_world is None
                else list(args.policy_initial_gate_position_world)
            ),
            "carry_prefix_gate_rates": args.carry_prefix_gate_rates,
            "prefix_max_action_error": prefix_max_action_error,
            "late_checkpoint": (
                None if args.late_checkpoint is None else str(args.late_checkpoint)
            ),
            "late_checkpoint_sha256": (
                None
                if args.late_checkpoint is None
                else sha256_file(args.late_checkpoint)
            ),
            "late_prefix_max_action_error": late_prefix_max_action_error,
            "switch_after_gate2_seconds": args.switch_after_gate2_seconds,
            "switch_until_gate2_seconds": args.switch_until_gate2_seconds,
            "activation_delay_checkpoint": (
                None
                if args.activation_delay_checkpoint is None
                else str(args.activation_delay_checkpoint)
            ),
            "activation_delay_checkpoint_sha256": (
                None
                if args.activation_delay_checkpoint is None
                else sha256_file(args.activation_delay_checkpoint)
            ),
            "activation_delay_prefix_max_action_error": (
                activation_delay_prefix_max_action_error),
            "gate2_activation_delay_seconds": args.gate2_activation_delay_seconds,
            "activation_delay_selected_steps": activation_delay_selected_steps,
            "schedule_action_semantics": (
                "single_puffer_checkpoint"
                if (
                    args.late_checkpoint is None
                    and args.activation_delay_checkpoint is None
                )
                else "whole_action_vector_selected_between_puffer_checkpoints"
            ),
            "late_selected_steps": late_selected_steps,
            "episodes": args.episodes,
            "seed": args.seed,
            "gate_radius": args.gate_radius,
            "perturbed": args.perturbed,
            "perturbation_scale": args.perturbation_scale,
            "perturbation_components": args.perturbation_components,
            "start_perturbation_dimensions": args.start_perturbation_dimensions,
            "steps": steps,
            "elapsed_seconds": round(time.time() - started, 6),
            "config_overrides": overrides,
            "dataset": dataset_metadata,
            "trace": trace_metadata,
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
