"""Training-only query for the admitted VQ2 alignment governor.

This module is never imported by the deployed actor.  It decodes the native
privileged tail only to label states visited by a student during DAgger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE, PRIVILEGED_SIZE


@dataclass(frozen=True)
class AlignmentOracleConfig:
    target_forward_speed_m_s: float = 2.0
    maximum_velocity_m_s: float = 20.0
    hover_thrust: float = 0.27
    minimum_thrust: float = 0.18
    maximum_thrust: float = 0.42
    vertical_accel_per_thrust: float = 32.81
    lateral_accel_scale: float = 0.107491
    gravity_m_s2: float = 8.69
    linear_drag: float = 0.14
    maximum_roll_rad: float = 0.50


def _quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors by scalar-first quaternions using float32 arithmetic."""

    scalar = q[:, :1]
    xyz = q[:, 1:]
    first = 2.0 * np.sum(xyz * vector, axis=1, keepdims=True) * xyz
    second = (scalar * scalar - np.sum(xyz * xyz, axis=1, keepdims=True)) * vector
    third = 2.0 * scalar * np.cross(xyz, vector)
    return (first + second + third).astype(np.float32)


def alignment_oracle_action(
    environment_observation: np.ndarray,
    config: AlignmentOracleConfig = AlignmentOracleConfig(),
) -> np.ndarray:
    """Return the four-channel SF009 label for a batch of current states."""

    observation = np.asarray(environment_observation)
    if observation.ndim != 2 or observation.shape[1] != ENV_OBS_SIZE:
        raise ValueError(
            f"oracle query requires [batch,{ENV_OBS_SIZE}] native observations"
        )
    privilege = observation[:, LEGAL_OBS_SIZE:]
    if privilege.shape[1] != PRIVILEGED_SIZE:
        raise RuntimeError("native privileged ABI changed")
    if not np.isfinite(privilege).all():
        raise RuntimeError("oracle query received non-finite privileged state")

    # Native fields 3:6 are tanh(relative_body / 10).  The admitted course
    # distribution never saturates these values; clipping only avoids atanh(1)
    # after the native observation's final [-1,1] safety clamp.
    relative_body = np.arctanh(
        np.clip(privilege[:, 3:6], -1.0 + 1e-7, 1.0 - 1e-7)
    ).astype(np.float32)
    relative_body *= np.float32(10.0)
    quaternion = privilege[:, 12:16].astype(np.float32, copy=False)
    relative_world = _quat_rotate(quaternion, relative_body)
    velocity = privilege[:, 6:9].astype(np.float32, copy=False)
    velocity = velocity * np.float32(config.maximum_velocity_m_s)

    radial_error = np.sqrt(
        relative_world[:, 1] * relative_world[:, 1]
        + relative_world[:, 2] * relative_world[:, 2]
    )
    alignment = np.clip(np.float32(1.0) - radial_error, 0.0, 1.0)
    approach_speed = np.float32(max(config.target_forward_speed_m_s, 0.20))
    target_forward = np.float32(0.20) + alignment * (
        approach_speed - np.float32(0.20)
    )
    pitch = np.clip(
        np.float32(0.45) * (velocity[:, 0] - target_forward), -0.80, 0.80
    )

    accel_y = np.float32(0.40) * relative_world[:, 1] - np.float32(
        1.20
    ) * velocity[:, 1]
    accel_z = np.float32(0.80) * relative_world[:, 2] - np.float32(
        1.50
    ) * velocity[:, 2]
    force_y = accel_y + np.float32(config.linear_drag) * velocity[:, 1]
    force_z = (
        np.float32(config.gravity_m_s2)
        + accel_z
        + np.float32(config.linear_drag) * velocity[:, 2]
    )
    qw, qx, qy, qz = (quaternion[:, index] for index in range(4))
    sin_pitch = np.float32(2.0) * (qw * qy - qz * qx)
    attitude_pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    desired_roll = np.arctan2(
        -force_y * np.cos(attitude_pitch),
        np.maximum(
            force_z * np.float32(config.lateral_accel_scale), np.float32(1e-3)
        ),
    )
    desired_roll = np.clip(
        desired_roll, -config.maximum_roll_rad, config.maximum_roll_rad
    )
    roll = desired_roll / np.float32(max(config.maximum_roll_rad, 1e-4))

    vertical_fraction = np.maximum(
        np.cos(desired_roll) * np.cos(attitude_pitch), np.float32(0.50)
    )
    command_thrust = force_z / np.maximum(
        np.float32(config.vertical_accel_per_thrust) * vertical_fraction,
        np.float32(1e-4),
    )
    above = command_thrust >= np.float32(config.hover_thrust)
    thrust = np.where(
        above,
        (command_thrust - np.float32(config.hover_thrust))
        / np.float32(max(config.maximum_thrust - config.hover_thrust, 1e-4)),
        (command_thrust - np.float32(config.hover_thrust))
        / np.float32(max(config.hover_thrust - config.minimum_thrust, 1e-4)),
    )
    action = np.stack(
        (
            np.clip(pitch, -0.80, 0.80),
            np.clip(roll, -1.0, 1.0),
            np.clip(thrust, -1.0, 1.0),
            np.zeros_like(pitch),
        ),
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(action).all():
        raise RuntimeError("oracle query produced a non-finite label")
    return action

