"""Offline contract helpers for the N294-prefix visual-suffix VQ2 composite.

The deployed policies remain separate and whole-output.  N294 consumes its
historical 32-value legal view while the visual suffix consumes the leading
4,118 legal visual values plus held public progress.  The native privileged
tail is used here only to reproduce N294's already-admitted camera/IMU view in
the command-free training simulator; it never enters either neural policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pufferlib.vq2_informed import (
    ACTION_HISTORY,
    ENV_OBS_SIZE,
    LEGAL_OBS_SIZE,
    PRIVILEGED_NAMES,
)
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE


LEGACY_OBS_SIZE = 32
ACTION_SIZE = 4
GATE_COUNT_SCALE = 6.0
CAMERA_HALF_FOV_RAD = np.float32(np.pi / 4.0)
GATE_INNER_WIDTH_M = np.float32(1.5)

_PRIVILEGED_OFFSET = LEGAL_OBS_SIZE
_PRIVILEGED_INDEX = {
    name: _PRIVILEGED_OFFSET + index
    for index, name in enumerate(PRIVILEGED_NAMES)
}


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate batched vectors by batched ``wxyz`` quaternions."""

    q = np.asarray(quaternion, dtype=np.float32)
    v = np.asarray(vector, dtype=np.float32)
    if q.shape != (v.shape[0], 4) or v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("quaternion/vector batches do not align")
    xyz = q[:, 1:4]
    cross = np.cross(xyz, v)
    return v + np.float32(2.0) * (
        q[:, 0:1] * cross + np.cross(xyz, cross)
    )


def _quat_inverse(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float32).copy()
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("quaternion batch must have shape [agents,4]")
    norm = np.sum(q * q, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 1e-12):
        raise ValueError("quaternion batch contains an invalid rotation")
    q[:, 1:4] *= np.float32(-1.0)
    return q / norm


def visual_suffix_observation(
    environment_observation: np.ndarray,
    held_phase: np.ndarray,
) -> np.ndarray:
    """Return exactly 4,119 legal/public values and discard privilege."""

    values = np.asarray(environment_observation, dtype=np.float32)
    phase = np.asarray(held_phase, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ENV_OBS_SIZE:
        raise ValueError("visual environment observation ABI changed")
    if phase.shape != (values.shape[0],) or not np.isfinite(phase).all():
        raise ValueError("held public phase does not align with observations")
    result = np.concatenate(
        (values[:, :LEGAL_OBS_SIZE], phase[:, None]), axis=1
    ).astype(np.float32, copy=False)
    if result.shape[1] != PHASE_LEGAL_OBS_SIZE:
        raise RuntimeError("visual suffix observation width changed")
    return result


def select_complete_actions(
    n294_action: np.ndarray,
    suffix_action: np.ndarray,
    held_phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one complete Puffer action vector using held official progress."""

    prefix = np.asarray(n294_action, dtype=np.float32)
    suffix = np.asarray(suffix_action, dtype=np.float32)
    phase = np.asarray(held_phase, dtype=np.float32)
    if prefix.shape != suffix.shape or prefix.ndim != 2 or prefix.shape[1] != ACTION_SIZE:
        raise ValueError("Puffer action batches must both have shape [agents,4]")
    if phase.shape != (prefix.shape[0],):
        raise ValueError("held public phase does not align with actions")
    suffix_selected = phase >= np.float32(1.0 / GATE_COUNT_SCALE) - np.float32(1e-7)
    selected = np.where(suffix_selected[:, None], suffix, prefix)
    return np.clip(selected, -1.0, 1.0).astype(np.float32), suffix_selected


@dataclass
class LegacyN294ObservationAdapter:
    """Stateful native reproduction of N294's frozen 32-value sensor view."""

    agents: int
    dt_s: float = 1.0 / 64.0
    time_limit_s: float = 14.0
    sample_interval_steps: int = 4
    dropout_range_m: float = 4.25
    dropout_from_gate_index: int = 0

    def __post_init__(self) -> None:
        if self.agents <= 0:
            raise ValueError("agents must be positive")
        if self.dt_s <= 0.0 or self.time_limit_s <= 0.0:
            raise ValueError("adapter timing must be positive")
        if self.sample_interval_steps <= 0:
            raise ValueError("sample interval must be positive")
        self.valid = np.zeros(self.agents, dtype=bool)
        self.gate_index = np.full(self.agents, -1, dtype=np.int32)
        self.raw_world = np.zeros((self.agents, 3), dtype=np.float32)
        self.filtered_world = np.zeros((self.agents, 3), dtype=np.float32)
        self.rate_world = np.zeros((self.agents, 3), dtype=np.float32)

    def reset(self) -> None:
        self.valid.fill(False)
        self.gate_index.fill(-1)
        self.raw_world.fill(0.0)
        self.filtered_world.fill(0.0)
        self.rate_world.fill(0.0)

    def observe(
        self,
        environment_observation: np.ndarray,
        raw_phase: np.ndarray,
        held_phase: np.ndarray,
        *,
        step: int,
    ) -> np.ndarray:
        values = np.asarray(environment_observation, dtype=np.float32)
        raw = np.asarray(raw_phase, dtype=np.float32)
        held = np.asarray(held_phase, dtype=np.float32)
        if values.shape != (self.agents, ENV_OBS_SIZE):
            raise ValueError("visual environment observation ABI changed")
        if raw.shape != (self.agents,) or held.shape != (self.agents,):
            raise ValueError("race phases do not align with adapter agents")
        if step < 0:
            raise ValueError("step must be nonnegative")

        result = np.zeros((self.agents, LEGACY_OBS_SIZE), dtype=np.float32)
        quaternion = values[
            :,
            _PRIVILEGED_INDEX["attitude_qw"] : _PRIVILEGED_INDEX["attitude_qz"] + 1,
        ]
        inverse = _quat_inverse(quaternion)
        relative_body = np.arctanh(
            np.clip(
                values[
                    :,
                    _PRIVILEGED_INDEX["gate_relative_body_x"]
                    : _PRIVILEGED_INDEX["gate_relative_body_z"] + 1,
                ],
                -0.999999,
                0.999999,
            )
        ) * np.float32(10.0)
        relative_world = _quat_rotate(quaternion, relative_body)
        current_gate = np.rint(raw * np.float32(GATE_COUNT_SCALE)).astype(np.int32)

        forward = relative_body[:, 0]
        right = relative_body[:, 1]
        down = -relative_body[:, 2]
        distance = np.linalg.norm(relative_body, axis=1)
        yaw_error = np.arctan2(right, np.maximum(forward, np.float32(1e-4)))
        elevation = np.arctan2(-down, np.maximum(forward, np.float32(1e-4)))
        physically_visible = (
            (forward > np.float32(0.05))
            & (np.abs(yaw_error) <= CAMERA_HALF_FOV_RAD)
            & (np.abs(elevation) <= CAMERA_HALF_FOV_RAD)
        )
        physically_visible &= ~(
            (current_gate >= self.dropout_from_gate_index)
            & (distance <= np.float32(self.dropout_range_m))
        )
        physically_visible &= step % self.sample_interval_steps == 0
        physically_visible &= self.valid | (distance <= np.float32(35.0))

        position_alpha = np.float32(1.0 - np.exp(-self.dt_s / 0.30))
        rate_alpha = np.float32(1.0 - np.exp(-self.dt_s / 0.60))
        innovation_limit = np.float32(max(3.0, 30.0 * self.dt_s))

        for agent in range(self.agents):
            if physically_visible[agent]:
                if not self.valid[agent] or self.gate_index[agent] != current_gate[agent]:
                    self.raw_world[agent] = relative_world[agent]
                    self.filtered_world[agent] = relative_world[agent]
                    self.rate_world[agent] = 0.0
                    self.valid[agent] = True
                    self.gate_index[agent] = current_gate[agent]
                else:
                    innovation = relative_world[agent] - self.raw_world[agent]
                    if np.linalg.norm(innovation) <= innovation_limit:
                        previous = self.filtered_world[agent].copy()
                        self.filtered_world[agent] = previous + position_alpha * (
                            relative_world[agent] - previous
                        )
                        measured_rate = (
                            self.filtered_world[agent] - previous
                        ) / np.float32(self.dt_s)
                        self.rate_world[agent] += rate_alpha * (
                            measured_rate - self.rate_world[agent]
                        )
                        self.raw_world[agent] = relative_world[agent]
            elif not self.valid[agent] or self.gate_index[agent] != current_gate[agent]:
                self.valid[agent] = False
                self.gate_index[agent] = -1
                self.rate_world[agent] = 0.0

        held_body = _quat_rotate(inverse, self.raw_world)
        rate_body = _quat_rotate(inverse, self.rate_world)
        visible = self.valid & (self.gate_index == current_gate)

        result[:, 3:6] = values[
            :,
            _PRIVILEGED_INDEX["body_rate_x"]
            : _PRIVILEGED_INDEX["body_rate_z"] + 1,
        ]
        result[:, 6:10] = quaternion
        result[:, 0] = np.tanh(rate_body[:, 0] * np.float32(0.2))
        result[:, 1] = np.tanh(rate_body[:, 1] * np.float32(1.0 / 3.0))
        result[:, 2] = np.tanh(-rate_body[:, 2] * np.float32(1.0 / 3.0))
        result[:, 10] = visible.astype(np.float32)

        held_forward = held_body[:, 0]
        held_right = held_body[:, 1]
        held_down = -held_body[:, 2]
        held_range = np.maximum(np.linalg.norm(held_body, axis=1), np.float32(1e-3))
        held_yaw = np.arctan2(
            held_right, np.maximum(held_forward, np.float32(1e-4))
        )
        held_elevation = np.arctan2(
            -held_down, np.maximum(held_forward, np.float32(1e-4))
        )
        result[visible, 11] = np.tanh(held_forward[visible] * np.float32(0.1))
        result[visible, 12] = np.tanh(held_right[visible] * np.float32(0.2))
        result[visible, 13] = np.tanh(held_down[visible] * np.float32(0.2))
        result[visible, 14] = held_yaw[visible] / CAMERA_HALF_FOV_RAD
        result[visible, 15] = held_elevation[visible] / CAMERA_HALF_FOV_RAD
        result[visible, 16] = np.clip(
            GATE_INNER_WIDTH_M / held_range[visible], 0.0, 1.0
        )
        yaw_score = np.float32(1.0) - np.abs(held_yaw) / CAMERA_HALF_FOV_RAD
        pitch_score = np.float32(1.0) - np.abs(held_elevation) / CAMERA_HALF_FOV_RAD
        result[visible, 17] = np.clip(
            np.float32(0.5) * (yaw_score[visible] + pitch_score[visible]),
            0.0,
            1.0,
        )
        result[:, 18] = np.clip(
            np.float32(step * self.dt_s / self.time_limit_s), 0.0, 1.0
        )
        result[:, 19:23] = values[
            :, ACTION_HISTORY.start : ACTION_HISTORY.start + ACTION_SIZE
        ]
        result[:, 23] = held
        held_gate = np.rint(held * np.float32(GATE_COUNT_SCALE)).astype(np.int32)
        for gate in range(6):
            result[:, 24 + gate] = (held_gate == gate).astype(np.float32)
        return np.clip(result, -1.0, 1.0).astype(np.float32)
