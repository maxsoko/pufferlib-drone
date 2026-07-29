#!/usr/bin/env python3
"""Coordinate-free six-gate phase composite for official policy-attitude control.

The callable preserves the live-proven recurrent prefix, resets independently
trained recurrent policies at active Gates 4--6, and uses only the deployed
32-value camera/IMU/progress observation for the two late observable adapters.
No gate coordinates or native vehicle state are available here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy


@dataclass
class _CompositeState:
    gate4: CheckpointPolicy
    gate5: CheckpointPolicy
    gate6: CheckpointPolicy
    previous_gate: int = -1
    gate5_counter_roll_latched: bool = False
    gate6_roll_direction: float = 0.0


_STATE: _CompositeState | None = None
_KEY: tuple[str, str, str, int, int, int, int, int, bool] | None = None
GATE1_SEVERE_DIRECT_RIGHT_M = -0.8


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _required_path(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    path = os.path.abspath(value)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _resolve() -> _CompositeState:
    global _KEY, _STATE
    dimensions = (
        int(os.getenv("PUFFER_POLICY_INPUT_DIM", "32")),
        int(os.getenv("PUFFER_POLICY_HIDDEN_DIM", "128")),
        int(os.getenv("PUFFER_POLICY_NUM_LAYERS", "3")),
        int(os.getenv("PUFFER_POLICY_NUM_ACTIONS", "4")),
        int(os.getenv("PUFFER_POLICY_LAYOUT_PRECISION_BYTES", "4")),
        _env_bool("PUFFER_POLICY_NATIVE_BF16", False),
    )
    key = (
        _required_path("PUFFER_POLICY_GATE4_CHECKPOINT_PATH"),
        _required_path("PUFFER_POLICY_GATE5_CHECKPOINT_PATH"),
        _required_path("PUFFER_POLICY_GATE6_CHECKPOINT_PATH"),
        *dimensions,
    )
    if _STATE is None or _KEY != key:
        load = dict(
            input_dim=dimensions[0],
            hidden_dim=dimensions[1],
            num_layers=dimensions[2],
            num_actions=dimensions[3],
            layout_precision_bytes=dimensions[4],
            native_bf16=dimensions[5],
        )
        _STATE = _CompositeState(
            gate4=CheckpointPolicy.load(key[0], **load),
            gate5=CheckpointPolicy.load(key[1], **load),
            gate6=CheckpointPolicy.load(key[2], **load),
        )
        _KEY = key
    return _STATE


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _unscale_tanh(value: float, scale: float) -> float:
    clipped = np.float32(np.clip(np.float32(value), -0.9999, 0.9999))
    return float(np.arctanh(clipped) / np.float32(scale))


def _inverse_tanh_norm(value: float, scale: float) -> float:
    """Decode the legacy live-prefix observation exactly."""

    return math.atanh(_clamp(float(value), -0.999, 0.999)) * scale


def _gate_index(observation: np.ndarray) -> int:
    scaled = float(observation[23]) * 6.0
    gate = int(round(scaled))
    if abs(scaled - gate) > 1e-3 or gate < 0 or gate > 6:
        raise ValueError(f"invalid six-gate progress observation: {scaled}")
    return gate


def _quat_normalized(observation: np.ndarray) -> np.ndarray:
    quat = np.asarray(observation[6:10], dtype=np.float32).copy()
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-6:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / np.float32(norm)


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    qv = quat[1:4]
    cross = np.float32(2.0) * np.cross(qv, vector)
    return vector + quat[0] * cross + np.cross(qv, cross)


def _roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = (float(value) for value in quat)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def _coordinate_free_final(
    observation: np.ndarray,
    actions: list[float],
    state: _CompositeState,
) -> list[float]:
    # Recurrent fallback stays active through detector dropout.
    actions[0] = _clamp(actions[0] + 0.15)
    actions[2] = _clamp(actions[2] - 0.10)
    if float(observation[10]) <= 0.5:
        return actions

    forward = _unscale_tanh(float(observation[11]), 0.1)
    right = _unscale_tanh(float(observation[12]), 0.2)
    down = _unscale_tanh(float(observation[13]), 0.2)
    forward_rate = _unscale_tanh(float(observation[0]), 0.2)
    right_rate = _unscale_tanh(float(observation[1]), 1.0 / 3.0)
    down_rate = _unscale_tanh(float(observation[2]), 1.0 / 3.0)

    quat = _quat_normalized(observation)
    relative_body = np.asarray([forward, right, -down], dtype=np.float32)
    gate_rate_body = np.asarray(
        [forward_rate, right_rate, -down_rate], dtype=np.float32
    )
    relative_world = _quat_rotate(quat, relative_body)
    velocity_world = -_quat_rotate(quat, gate_rate_body)
    forward_speed = max(float(velocity_world[0]), 4.1875)
    time_to_plane = max(float(relative_world[0]) / forward_speed, 0.5)
    desired_vertical_accel = 2.0 * (
        float(relative_world[2]) - float(velocity_world[2]) * time_to_plane
    ) / (time_to_plane * time_to_plane)
    vertical_force = 8.69 + desired_vertical_accel + 0.14 * float(velocity_world[2])
    roll, pitch = _roll_pitch(quat)
    vertical_fraction = max(float(np.cos(roll) * np.cos(pitch)), 0.5)
    commanded_thrust = vertical_force / (32.81 * vertical_fraction)
    if commanded_thrust >= 0.27:
        actions[2] = _clamp((commanded_thrust - 0.27) / (0.42 - 0.27))
    else:
        actions[2] = _clamp((commanded_thrust - 0.27) / (0.27 - 0.18))

    if state.gate6_roll_direction == 0.0:
        state.gate6_roll_direction = 1.0 if right >= 0.0 else -1.0
    nominal_roll = -1.0 if forward > 2.0 else 0.5
    actions[1] = state.gate6_roll_direction * nominal_roll

    observable_speed = max(-forward_rate, 4.1875)
    actions[0] = _clamp(0.24 * _clamp((observable_speed - 5.0) * 0.35, -0.60, 0.60))
    actions[3] = 0.0
    return actions


def _promoted_gate3_roll_command(
    observation: np.ndarray,
) -> tuple[str, float | None]:
    """Return the frozen Candidate-128/129 roll mode and normalized command.

    The law passed the pre-registered 8/10 official reliability screen. It is
    stateless and changes only roll while official Gate 3 is active. All values
    below are the frozen promoted configuration; no live-result fitting occurs.
    """

    if float(observation[10]) < 0.5:
        return "base", None
    forward = _inverse_tanh_norm(float(observation[11]), 10.0)
    right = _inverse_tanh_norm(float(observation[12]), 5.0)
    forward_rate = _inverse_tanh_norm(float(observation[0]), 5.0)
    right_rate = _inverse_tanh_norm(float(observation[1]), 3.0)
    closing_rate = -forward_rate
    predicted_right = right
    if closing_rate >= 1.0:
        predicted_right += right_rate * max(forward - 2.0, 0.0) / closing_rate

    # Candidate 125's continuous severe-miss taper, retained by 128/129.
    adaptive_bank = (
        0.0 < forward <= 22.0
        and closing_rate >= 5.0
        and right < -0.3
        and predicted_right < -0.8
    )
    if adaptive_bank:
        severity = _clamp((-0.8 - predicted_right) / 2.7, 0.0, 1.0)
        roll_floor_rad = 0.45 + 0.05 * severity
        return "adaptive", roll_floor_rad / 0.5

    # Candidate 128's only change: start the 20--15 m bank at 4.0 m/s.
    early_bank = (
        15.0 < forward <= 20.0
        and closing_rate >= 4.0
        and right < -0.3
    )
    if early_bank:
        return "early", 0.45 / 0.5

    # Candidate 098's retained close intercept and full counter-bank.
    if 0.0 < forward <= 15.0:
        correction_needed = right < -0.3 and predicted_right < -0.8
        if correction_needed:
            return "close", 0.45 / 0.5
        return "counter", -1.0
    return "base", None


def _promoted_gate3_intercept(
    observation: np.ndarray,
    actions: list[float],
) -> list[float]:
    mode, roll_command = _promoted_gate3_roll_command(observation)
    if mode in {"adaptive", "early", "close"}:
        assert roll_command is not None
        actions[1] = max(actions[1], roll_command)
    elif mode == "counter":
        actions[1] = -1.0
    return actions


def _promoted_gate1_positive_intercept(
    observation: np.ndarray,
    actions: list[float],
) -> list[float]:
    """Reuse only the frozen positive intercept floors for official Gate 1.

    The Gate-3 counter-bank is deliberately phase-specific and must not be
    transferred. Candidate 006 also proved that Gate 3's permissive ``-0.3 m``
    trigger can saturate roll on a moderate Gate-1 offset and is unstable at
    rounded trace precision. Require a directly observed severe miss while
    retaining the recurrent action everywhere else.
    """

    mode, roll_command = _promoted_gate3_roll_command(observation)
    direct_right_m = _inverse_tanh_norm(float(observation[12]), 5.0)
    if (
        mode in {"adaptive", "early", "close"}
        and direct_right_m < GATE1_SEVERE_DIRECT_RIGHT_M
    ):
        assert roll_command is not None
        actions[1] = max(actions[1], roll_command)
    return actions


def reset() -> None:
    state = _resolve()
    state.gate4.reset_state()
    state.gate5.reset_state()
    state.gate6.reset_state()
    state.previous_gate = -1
    state.gate5_counter_roll_latched = False
    state.gate6_roll_direction = 0.0


def infer(observation: Sequence[float]) -> list[float]:
    state = _resolve()
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    gate = _gate_index(values)
    if gate <= 3:
        selected = state.gate4
    elif gate == 4:
        selected = state.gate5
    else:
        selected = state.gate6

    if gate != state.previous_gate:
        if 3 <= gate <= 5:
            selected.reset_state()
        if gate != 5:
            state.gate6_roll_direction = 0.0
        state.previous_gate = gate

    actions = selected.infer(values)
    if gate == 0:
        actions = _promoted_gate1_positive_intercept(values, actions)
    elif gate == 2:
        actions = _promoted_gate3_intercept(values, actions)
    elif gate == 4:
        actions[0] = _clamp(actions[0] + 0.10)
        if (
            not state.gate5_counter_roll_latched
            and float(values[10]) > 0.5
        ):
            forward = _unscale_tanh(float(values[11]), 0.1)
            if 0.0 < forward <= 10.0:
                state.gate5_counter_roll_latched = True
        if state.gate5_counter_roll_latched:
            actions[1] = -0.5
    elif gate == 5:
        actions = _coordinate_free_final(values, actions, state)
    return [_clamp(value) for value in actions]
