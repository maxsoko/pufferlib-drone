#!/usr/bin/env python3
"""Promoted 23-input prefix with the frozen 32-input six-gate tail.

Gates 1--3 use the v6c recurrent policy that passed official Gates 1 and 2 in
10/10 independent resets and Gate 3 in 8/10.  Gates 4--6 use the independently
reset N145 tail checkpoints and observable adapters. Input slots 30/31 carry
the legacy gate-pose confidence and 10.5-second elapsed fraction for the prefix
only. The current runner's field 22 already carries the previous normalized
yaw action expected by the recurrent checkpoint, so the prefix preserves it
unchanged. Both reserved slots are cleared before any 32-input tail checkpoint
is evaluated. A Gate-1-only late residual floor retains a strong negative roll
inside 6 m when the observable projection still places the opening more than
0.8 m to camera-right; it changes no other action and leaves stronger negative
roll untouched. A Gate-2 projected-aperture governor starts braking inside 12 m
when observable position/rate predict an edge strike, then adds the existing
bounded recentering law inside 6 m. If a positive projected miss remains above
0.65 m inside 4 m, a one-sided roll floor prevents the derivative term from
weakening the correction before the plane. A stateless Gate-2 vertical guard
applies a moderate thrust floor only while the current plane projection is
more than 2.5 m below gate center. It does not latch after recovery: a stronger
old floor produced a high contact, while full retirement produced a low
contact. Gate 3 is an exact behavioral rollback to the N160 live path that
passed the gate: apply the promoted intercept, then the close-severe boost with
its original 4 m counter-suppression bound. Later terminal-margin, vertical,
terminal-level, late-residual, and minimum-energy helpers remain available for
source-hashed historical replay but are retired from live inference. This
restores the recurrent policy's pitch, thrust, and yaw as well as N160's full
counter-bank/positive-bank/level roll sequence without another correction
layer.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_six_gate_composite as tail
from policy_callable_checkpoint import CheckpointPolicy


EXPECTED_TAIL_SOURCE_SHA256 = (
    "9eaab39694f67fdae6ebb6be52af9a282974e6778822fd84cd194ac5ade5d2a3"
)
_TAIL_SOURCE_PATH = Path(tail.__file__).resolve()
if hashlib.sha256(_TAIL_SOURCE_PATH.read_bytes()).hexdigest() != EXPECTED_TAIL_SOURCE_SHA256:
    raise RuntimeError("six-gate tail source hash mismatch")


_PREFIX: CheckpointPolicy | None = None
_PREFIX_KEY: tuple[str, int, int, int] | None = None
GATE1_LATE_RESIDUAL_MAX_FORWARD_M = 6.0
GATE1_LATE_RESIDUAL_PROJECTED_RIGHT_TRIGGER_M = 0.8
GATE1_LATE_RESIDUAL_ROLL_FLOOR_NORM = -0.7
GATE3_CLOSE_SEVERE_FORWARD_M = 4.0
GATE3_COUNTER_SUPPRESSION_FORWARD_M = 4.0
GATE3_CLOSE_COUNTER_ROLL_FLOOR = 0.0
GATE3_TERMINAL_MARGIN_FORWARD_M = 7.0
GATE3_TERMINAL_MARGIN_MIN_FORWARD_M = 4.0
GATE3_TERMINAL_MARGIN_PROJECTED_RIGHT_TRIGGER_M = -0.5
GATE3_TERMINAL_MARGIN_ROLL_FLOOR_NORM = 0.3
GATE3_MINIMUM_ENERGY_MAX_FORWARD_M = 6.5
GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M = 4.0
GATE3_MINIMUM_ENERGY_TARGET_RIGHT_M = -1.46
GATE3_MINIMUM_ENERGY_TARGET_RIGHT_RATE_M_S = 2.34
GATE3_MINIMUM_ENERGY_MIN_CONFIDENCE = 0.1
GATE3_MINIMUM_ENERGY_MIN_HORIZON_S = 0.12
GATE3_MINIMUM_ENERGY_GRAVITY_M_S2 = 9.80665
GATE3_MINIMUM_ENERGY_MAX_ROLL_RAD = 0.25
GATE3_POLICY_MAX_ROLL_RAD = 0.5
GATE3_LATE_POSITIVE_RESIDUAL_MAX_FORWARD_M = 4.0
GATE3_LATE_POSITIVE_RESIDUAL_MIN_CONFIDENCE = 0.15
GATE3_LATE_POSITIVE_RESIDUAL_MIN_RIGHT_M = 0.0
GATE3_LATE_POSITIVE_RESIDUAL_MIN_RIGHT_RATE_M_S = 2.0
GATE3_LATE_POSITIVE_RESIDUAL_PROJECTED_RIGHT_TRIGGER_M = 0.5
GATE3_LATE_POSITIVE_RESIDUAL_ROLL_FLOOR_NORM = -1.0
GATE3_VERTICAL_FLOOR_FORWARD_M = 12.0
GATE3_VERTICAL_FLOOR_PROJECTED_DOWN_TRIGGER_M = -1.0
GATE3_VERTICAL_FLOOR_THRUST_NORM = 0.0
GATE3_TERMINAL_LEVEL_FORWARD_M = 6.0
GATE3_TERMINAL_MAX_ABS_PROJECTED_RIGHT_M = 0.5
GATE3_TERMINAL_MAX_ABS_PROJECTED_DOWN_M = 1.0
GATE2_EARLY_BRAKE_FORWARD_M = 12.0
GATE2_GOVERNOR_FORWARD_M = 6.0
GATE2_PROJECTED_MISS_THRESHOLD_M = 0.3
GATE2_INTERCEPT_SPEED_FLOOR_M_S = 3.0
GATE2_ROLL_KP = 0.6
GATE2_ROLL_KD = 0.35
GATE2_ROLL_LIMIT = 0.7
GATE2_BRAKE_PITCH_NORM = -0.2
GATE2_LATE_RESIDUAL_MAX_FORWARD_M = 4.0
GATE2_LATE_RESIDUAL_PROJECTED_RIGHT_TRIGGER_M = 0.65
GATE2_LATE_RESIDUAL_ROLL_FLOOR_NORM = -0.5
GATE2_VERTICAL_FLOOR_FORWARD_M = 12.0
GATE2_VERTICAL_PROJECTED_DOWN_TRIGGER_M = -2.5
GATE2_VERTICAL_THRUST_FLOOR_NORM = 0.3
GATE2_VERTICAL_FLOOR_RETIRED = False
GATE2_VERTICAL_FLOOR_STATELESS = True
GATE4_MAX_INITIAL_RANGE_M = 35.0
GATE4_ASSOCIATION_GRACE_S = 1.0
GATE4_MAX_POSE_JUMP_M = 8.0
GATE4_MAX_ANCHOR_DRIFT_M = 12.0
GATE4_REANCHOR_PERSISTENCE_S = 0.125
GATE4_REANCHOR_MAX_FORWARD_M = 20.75
GATE4_REANCHOR_MAX_ABS_RIGHT_M = 4.5
GATE4_REANCHOR_MIN_DOWN_M = 1.5
GATE4_REANCHOR_MAX_COUNT = 2
GATE4_INITIAL_MAX_ABS_RIGHT_M = 4.5
GATE4_LATERAL_ROLL_KP = 0.02
GATE4_MAX_LATERAL_ROLL_RAD = 0.10
GATE4_MAX_PITCH_RAD = 0.12
GATE4_ALIGNED_FORWARD_PITCH_RAD = 0.06
GATE4_CENTER_YAW_ERROR_RAD = 0.12
GATE4_CENTER_DOWN_ERROR_M = 1.5
GATE4_ACQUISITION_YAW_DELAY_S = 1.0
GATE4_ACQUISITION_MAX_YAW_STEP_RAD = 0.35
GATE4_DROPOUT_YAW_CARRY_S = 0.5
GATE4_HOVER_THRUST = 0.27
GATE4_MIN_THRUST = 0.18
GATE4_MAX_THRUST = 0.42
GATE4_VERTICAL_THRUST_KP = 0.02
GATE4_VERTICAL_CONTROL_DELAY_S = 1.0
GATE4_DESCENT_DELAY_S = 3.0
GATE4_DESCENT_THRUST = 0.22

_GATE4_PHASE_STARTED_S: float | None = None
_GATE4_LAST_VECTOR: tuple[float, float, float] | None = None
_GATE4_ANCHOR_VECTOR: tuple[float, float, float] | None = None
_GATE4_REJECTED_CLUSTER_STARTED_S: float | None = None
_GATE4_REJECTED_CLUSTER_VECTOR: tuple[float, float, float] | None = None
_GATE4_REANCHOR_COUNT = 0
_GATE4_LAST_VISIBLE_YAW_TARGET_RAD: float | None = None
_GATE4_LAST_VISIBLE_YAW_TARGET_S: float | None = None
_GATE2_GOVERNOR_LATCHED = False
_GATE2_VERTICAL_FLOOR_LATCHED = False
_GATE3_VERTICAL_FLOOR_LATCHED = False
_GATE3_TERMINAL_LEVEL_LATCHED = False
_GATE3_MINIMUM_ENERGY_ROLL_NORM: float | None = None
_GATE3_MINIMUM_ENERGY_LEVEL_LATCHED = False


def _required_path(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    path = os.path.abspath(value)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _resolve_prefix() -> CheckpointPolicy:
    global _PREFIX, _PREFIX_KEY
    key = (
        _required_path("PUFFER_POLICY_PREFIX_CHECKPOINT_PATH"),
        int(os.getenv("PUFFER_POLICY_HIDDEN_DIM", "128")),
        int(os.getenv("PUFFER_POLICY_NUM_LAYERS", "3")),
        int(os.getenv("PUFFER_POLICY_NUM_ACTIONS", "4")),
    )
    if _PREFIX is None or _PREFIX_KEY != key:
        _PREFIX = CheckpointPolicy.load(
            key[0],
            input_dim=23,
            hidden_dim=key[1],
            num_layers=key[2],
            num_actions=key[3],
            native_bf16=False,
            layout_precision_bytes=2,
        )
        _PREFIX_KEY = key
    return _PREFIX


def _legacy_prefix_observation(values: np.ndarray) -> np.ndarray:
    legacy = np.asarray(values[:23], dtype=np.float32).copy()
    legacy[17] = np.float32(values[30])
    legacy[18] = np.float32(values[31])
    return legacy


def _tail_observation(values: np.ndarray) -> np.ndarray:
    current = np.asarray(values, dtype=np.float32).copy()
    current[30:32] = np.float32(0.0)
    return current


def _gate1_late_residual_roll_floor(
    values: np.ndarray, actions: list[float]
) -> list[float]:
    """Retain rightward correction for a late positive Gate-1 miss."""

    if float(values[10]) < 0.5:
        return actions
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    if not 0.0 < forward_m <= GATE1_LATE_RESIDUAL_MAX_FORWARD_M:
        return actions
    forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
    right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
    right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
    time_to_plane_s = forward_m / max(
        -forward_rate_m_s,
        GATE2_INTERCEPT_SPEED_FLOOR_M_S,
    )
    projected_right_m = right_m + right_rate_m_s * time_to_plane_s
    if projected_right_m > GATE1_LATE_RESIDUAL_PROJECTED_RIGHT_TRIGGER_M:
        actions[1] = min(
            float(actions[1]),
            GATE1_LATE_RESIDUAL_ROLL_FLOOR_NORM,
        )
    return actions


def _gate3_close_severe_boost(
    values: np.ndarray, actions: list[float]
) -> list[float]:
    """Keep the promoted Gate-3 crossing bank one-sided inside four metres."""

    mode, _roll_command = tail._promoted_gate3_roll_command(values)
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    if not 0.0 < forward_m:
        return actions
    if mode == "adaptive" and forward_m <= GATE3_CLOSE_SEVERE_FORWARD_M:
        actions[1] = 1.0
    elif mode == "counter" and forward_m <= GATE3_COUNTER_SUPPRESSION_FORWARD_M:
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        if right_m < 0.0:
            actions[1] = max(actions[1], GATE3_CLOSE_COUNTER_ROLL_FLOOR)
    return actions


def _gate3_terminal_level(
    values: np.ndarray, actions: list[float]
) -> list[float]:
    """Level roll after Gate 3 projects safely through its aperture."""

    global _GATE3_TERMINAL_LEVEL_LATCHED
    if not _GATE3_TERMINAL_LEVEL_LATCHED and float(values[10]) >= 0.5:
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
        closing_m_s = -forward_rate_m_s
        if (
            0.0 < forward_m <= GATE3_TERMINAL_LEVEL_FORWARD_M
            and closing_m_s >= 1.0
        ):
            time_to_terminal_s = max(forward_m - 2.0, 0.0) / closing_m_s
            right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
            down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
            right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
            down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
            projected_right_m = right_m + right_rate_m_s * time_to_terminal_s
            projected_down_m = down_m + down_rate_m_s * time_to_terminal_s
            if (
                abs(projected_right_m)
                <= GATE3_TERMINAL_MAX_ABS_PROJECTED_RIGHT_M
                and abs(projected_down_m)
                <= GATE3_TERMINAL_MAX_ABS_PROJECTED_DOWN_M
            ):
                _GATE3_TERMINAL_LEVEL_LATCHED = True
    if not _GATE3_TERMINAL_LEVEL_LATCHED:
        return actions
    actions[1] = 0.0
    return actions


def _gate3_terminal_margin_boost(
    values: np.ndarray, actions: list[float]
) -> list[float]:
    """Retain a small left correction until the terminal margin is safe."""

    if float(values[10]) < 0.5:
        return actions
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
    if not (
        GATE3_TERMINAL_MARGIN_MIN_FORWARD_M
        < forward_m
        <= GATE3_TERMINAL_MARGIN_FORWARD_M
        and closing_m_s >= 1.0
    ):
        return actions
    right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
    right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
    time_to_terminal_s = max(forward_m - 2.0, 0.0) / closing_m_s
    projected_right_m = right_m + right_rate_m_s * time_to_terminal_s
    if projected_right_m < GATE3_TERMINAL_MARGIN_PROJECTED_RIGHT_TRIGGER_M:
        actions[1] = max(
            float(actions[1]),
            GATE3_TERMINAL_MARGIN_ROLL_FLOOR_NORM,
        )
    return actions


def _gate3_minimum_energy_lateral_controller(
    values: np.ndarray, actions: list[float]
) -> list[float]:
    """Commit once to the clean Gate-3 handoff, then latch level."""

    global _GATE3_MINIMUM_ENERGY_ROLL_NORM
    global _GATE3_MINIMUM_ENERGY_LEVEL_LATCHED

    if _GATE3_MINIMUM_ENERGY_LEVEL_LATCHED:
        actions[1] = 0.0
        return actions
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
    if (
        _GATE3_MINIMUM_ENERGY_ROLL_NORM is not None
        and 0.0 < forward_m <= GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M
        and closing_m_s >= 1.0
    ):
        _GATE3_MINIMUM_ENERGY_LEVEL_LATCHED = True
        actions[1] = 0.0
        return actions
    if _GATE3_MINIMUM_ENERGY_ROLL_NORM is not None:
        actions[1] = _GATE3_MINIMUM_ENERGY_ROLL_NORM
        return actions
    if not (
        float(values[10]) >= 0.5
        and float(values[30]) >= GATE3_MINIMUM_ENERGY_MIN_CONFIDENCE
        and GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M
        < forward_m
        <= GATE3_MINIMUM_ENERGY_MAX_FORWARD_M
        and closing_m_s >= 1.0
    ):
        return actions

    right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
    right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
    horizon_s = max(
        (forward_m - GATE3_MINIMUM_ENERGY_HANDOFF_FORWARD_M) / closing_m_s,
        GATE3_MINIMUM_ENERGY_MIN_HORIZON_S,
    )
    # Exact first control of the minimum-integral-a^2 double-integrator
    # trajectory satisfying r(T)=r_target and v(T)=v_target.
    lateral_accel_m_s2 = (
        6.0 * (GATE3_MINIMUM_ENERGY_TARGET_RIGHT_M - right_m) / horizon_s**2
        - (
            4.0 * right_rate_m_s
            + 2.0 * GATE3_MINIMUM_ENERGY_TARGET_RIGHT_RATE_M_S
        )
        / horizon_s
    )
    roll_rad = math.atan(
        lateral_accel_m_s2 / GATE3_MINIMUM_ENERGY_GRAVITY_M_S2
    )
    roll_rad = tail._clamp(
        roll_rad,
        -GATE3_MINIMUM_ENERGY_MAX_ROLL_RAD,
        GATE3_MINIMUM_ENERGY_MAX_ROLL_RAD,
    )
    _GATE3_MINIMUM_ENERGY_ROLL_NORM = tail._clamp(
        roll_rad / GATE3_POLICY_MAX_ROLL_RAD,
        -1.0,
        1.0,
    )
    actions[1] = _GATE3_MINIMUM_ENERGY_ROLL_NORM
    return actions


def _gate3_late_positive_residual_roll_floor(
    values: np.ndarray, actions: list[float]
) -> list[float]:
    """Counter a trustworthy late Gate-3 overshoot without using stale pose."""

    if (
        float(values[10]) < 0.5
        or float(values[30]) < GATE3_LATE_POSITIVE_RESIDUAL_MIN_CONFIDENCE
    ):
        return actions
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
    if not (
        0.0 < forward_m <= GATE3_LATE_POSITIVE_RESIDUAL_MAX_FORWARD_M
        and closing_m_s >= 1.0
    ):
        return actions
    right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
    right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
    time_to_terminal_s = max(forward_m - 2.0, 0.0) / closing_m_s
    projected_right_m = right_m + right_rate_m_s * time_to_terminal_s
    if (
        right_m >= GATE3_LATE_POSITIVE_RESIDUAL_MIN_RIGHT_M
        and right_rate_m_s >= GATE3_LATE_POSITIVE_RESIDUAL_MIN_RIGHT_RATE_M_S
        and projected_right_m
        > GATE3_LATE_POSITIVE_RESIDUAL_PROJECTED_RIGHT_TRIGGER_M
    ):
        actions[1] = min(
            float(actions[1]),
            GATE3_LATE_POSITIVE_RESIDUAL_ROLL_FLOOR_NORM,
        )
    return actions


def _gate3_projected_vertical_floor(
    values: np.ndarray, actions: list[float]
) -> list[float]:
    """Floor low thrust only when Gate 3 projects above the vehicle."""

    global _GATE3_VERTICAL_FLOOR_LATCHED
    if not _GATE3_VERTICAL_FLOOR_LATCHED and float(values[10]) >= 0.5:
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        if (
            0.0 < forward_m <= GATE3_VERTICAL_FLOOR_FORWARD_M
            and closing_m_s >= 1.0
        ):
            time_to_terminal_s = max(forward_m - 2.0, 0.0) / closing_m_s
            down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
            down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
            projected_down_m = down_m + down_rate_m_s * time_to_terminal_s
            if projected_down_m < GATE3_VERTICAL_FLOOR_PROJECTED_DOWN_TRIGGER_M:
                _GATE3_VERTICAL_FLOOR_LATCHED = True
    if _GATE3_VERTICAL_FLOOR_LATCHED:
        actions[2] = max(
            float(actions[2]),
            GATE3_VERTICAL_FLOOR_THRUST_NORM,
        )
    return actions


def _clear_gate3_state() -> None:
    global _GATE3_VERTICAL_FLOOR_LATCHED
    global _GATE3_TERMINAL_LEVEL_LATCHED
    global _GATE3_MINIMUM_ENERGY_ROLL_NORM
    global _GATE3_MINIMUM_ENERGY_LEVEL_LATCHED
    _GATE3_VERTICAL_FLOOR_LATCHED = False
    _GATE3_TERMINAL_LEVEL_LATCHED = False
    _GATE3_MINIMUM_ENERGY_ROLL_NORM = None
    _GATE3_MINIMUM_ENERGY_LEVEL_LATCHED = False


def _gate2_projected_aperture_governor(
    values: np.ndarray, actions: list[float]
) -> list[float]:
    """Brake early, then recenter close, when Gate 2 projects outside."""

    global _GATE2_GOVERNOR_LATCHED
    if float(values[10]) < 0.5:
        return actions
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    if not 0.0 < forward_m <= GATE2_EARLY_BRAKE_FORWARD_M:
        return actions
    right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
    forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
    right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
    time_to_plane_s = forward_m / max(
        -forward_rate_m_s, GATE2_INTERCEPT_SPEED_FLOOR_M_S
    )
    projected_right_m = right_m + right_rate_m_s * time_to_plane_s
    if abs(projected_right_m) > GATE2_PROJECTED_MISS_THRESHOLD_M:
        _GATE2_GOVERNOR_LATCHED = True
    if not _GATE2_GOVERNOR_LATCHED:
        return actions

    actions[0] = min(float(actions[0]), GATE2_BRAKE_PITCH_NORM)
    if forward_m > GATE2_GOVERNOR_FORWARD_M:
        return actions
    actions[1] = tail._clamp(
        -GATE2_ROLL_KP * right_m - GATE2_ROLL_KD * right_rate_m_s,
        -GATE2_ROLL_LIMIT,
        GATE2_ROLL_LIMIT,
    )
    return actions


def _gate2_projected_vertical_floor(
    values: np.ndarray, actions: list[float]
) -> list[float]:
    """Moderately floor thrust only while the current Gate-2 miss is low."""

    global _GATE2_VERTICAL_FLOOR_LATCHED
    _GATE2_VERTICAL_FLOOR_LATCHED = False
    if float(values[10]) < 0.5:
        return actions
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    if not 0.0 < forward_m <= GATE2_VERTICAL_FLOOR_FORWARD_M:
        return actions
    forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
    time_to_plane_s = forward_m / max(
        -forward_rate_m_s,
        GATE2_INTERCEPT_SPEED_FLOOR_M_S,
    )
    down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
    down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
    projected_down_m = down_m + down_rate_m_s * time_to_plane_s
    if projected_down_m < GATE2_VERTICAL_PROJECTED_DOWN_TRIGGER_M:
        actions[2] = max(float(actions[2]), GATE2_VERTICAL_THRUST_FLOOR_NORM)
    return actions


def _gate2_late_residual_roll_floor(
    values: np.ndarray, actions: list[float]
) -> list[float]:
    """Retain rightward correction for a late positive projected miss."""

    if float(values[10]) < 0.5:
        return actions
    forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
    if not 0.0 < forward_m <= GATE2_LATE_RESIDUAL_MAX_FORWARD_M:
        return actions
    forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
    right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
    right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
    time_to_plane_s = forward_m / max(
        -forward_rate_m_s,
        GATE2_INTERCEPT_SPEED_FLOOR_M_S,
    )
    projected_right_m = right_m + right_rate_m_s * time_to_plane_s
    if projected_right_m > GATE2_LATE_RESIDUAL_PROJECTED_RIGHT_TRIGGER_M:
        actions[1] = min(
            float(actions[1]),
            GATE2_LATE_RESIDUAL_ROLL_FLOOR_NORM,
        )
    return actions


def _clear_gate2_state() -> None:
    global _GATE2_GOVERNOR_LATCHED
    global _GATE2_VERTICAL_FLOOR_LATCHED
    _GATE2_GOVERNOR_LATCHED = False
    _GATE2_VERTICAL_FLOOR_LATCHED = False


def _distance_m(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(left[index]) - float(right[index])) ** 2 for index in range(3))
    )


def _gate4_vector(values: np.ndarray) -> tuple[float, float, float] | None:
    if float(values[10]) < 0.5:
        return None
    vector = (
        tail._inverse_tanh_norm(float(values[11]), 10.0),
        tail._inverse_tanh_norm(float(values[12]), 5.0),
        tail._inverse_tanh_norm(float(values[13]), 5.0),
    )
    if not 0.0 < vector[0] <= GATE4_MAX_INITIAL_RANGE_M:
        return None
    return vector


def _clear_gate4_state() -> None:
    global _GATE4_PHASE_STARTED_S, _GATE4_LAST_VECTOR, _GATE4_ANCHOR_VECTOR
    global _GATE4_REJECTED_CLUSTER_STARTED_S, _GATE4_REJECTED_CLUSTER_VECTOR
    global _GATE4_REANCHOR_COUNT
    global _GATE4_LAST_VISIBLE_YAW_TARGET_RAD, _GATE4_LAST_VISIBLE_YAW_TARGET_S
    _GATE4_PHASE_STARTED_S = None
    _GATE4_LAST_VECTOR = None
    _GATE4_ANCHOR_VECTOR = None
    _GATE4_REJECTED_CLUSTER_STARTED_S = None
    _GATE4_REJECTED_CLUSTER_VECTOR = None
    _GATE4_REANCHOR_COUNT = 0
    _GATE4_LAST_VISIBLE_YAW_TARGET_RAD = None
    _GATE4_LAST_VISIBLE_YAW_TARGET_S = None


def _gate4_current_association(
    values: np.ndarray,
    now_s: float,
) -> tuple[tuple[float, float, float] | None, float]:
    """Return only a current, coherent raw Gate-4 pose and phase elapsed time."""

    global _GATE4_PHASE_STARTED_S, _GATE4_LAST_VECTOR, _GATE4_ANCHOR_VECTOR
    global _GATE4_REJECTED_CLUSTER_STARTED_S, _GATE4_REJECTED_CLUSTER_VECTOR
    global _GATE4_REANCHOR_COUNT

    if _GATE4_PHASE_STARTED_S is None:
        _GATE4_PHASE_STARTED_S = now_s
    elapsed_s = max(0.0, now_s - _GATE4_PHASE_STARTED_S)
    vector = _gate4_vector(values)
    if vector is None:
        _GATE4_REJECTED_CLUSTER_STARTED_S = None
        _GATE4_REJECTED_CLUSTER_VECTOR = None
        return None, elapsed_s

    # Candidate 018 proved that the transition grace can still see the
    # just-crossed/far-side frame at more than 10 m lateral displacement. Do
    # not let that family become the initial Gate-4 anchor; use the same
    # lateral aperture bound already required for coherent close re-anchors.
    if (
        _GATE4_ANCHOR_VECTOR is None
        and abs(vector[1]) > GATE4_INITIAL_MAX_ABS_RIGHT_M
    ):
        _GATE4_REJECTED_CLUSTER_STARTED_S = None
        _GATE4_REJECTED_CLUSTER_VECTOR = None
        return None, elapsed_s

    in_grace = elapsed_s <= GATE4_ASSOCIATION_GRACE_S
    associated = in_grace or (
        (_GATE4_LAST_VECTOR is None or _distance_m(vector, _GATE4_LAST_VECTOR) <= GATE4_MAX_POSE_JUMP_M)
        and (
            _GATE4_ANCHOR_VECTOR is None
            or _distance_m(vector, _GATE4_ANCHOR_VECTOR)
            <= GATE4_MAX_ANCHOR_DRIFT_M
        )
    )
    if associated:
        _GATE4_LAST_VECTOR = vector
        if in_grace or _GATE4_ANCHOR_VECTOR is None:
            _GATE4_ANCHOR_VECTOR = vector
        _GATE4_REJECTED_CLUSTER_STARTED_S = None
        _GATE4_REJECTED_CLUSTER_VECTOR = None
        return vector, elapsed_s

    reanchor_eligible = (
        elapsed_s >= GATE4_DESCENT_DELAY_S
        and _GATE4_REANCHOR_COUNT < GATE4_REANCHOR_MAX_COUNT
        and vector[0] <= GATE4_REANCHOR_MAX_FORWARD_M
        and abs(vector[1]) <= GATE4_REANCHOR_MAX_ABS_RIGHT_M
        and vector[2] > GATE4_REANCHOR_MIN_DOWN_M
    )
    if not reanchor_eligible:
        _GATE4_REJECTED_CLUSTER_STARTED_S = None
        _GATE4_REJECTED_CLUSTER_VECTOR = None
        return None, elapsed_s

    if (
        _GATE4_REJECTED_CLUSTER_STARTED_S is None
        or _GATE4_REJECTED_CLUSTER_VECTOR is None
        or _distance_m(vector, _GATE4_REJECTED_CLUSTER_VECTOR)
        > GATE4_MAX_POSE_JUMP_M
    ):
        _GATE4_REJECTED_CLUSTER_STARTED_S = now_s
        _GATE4_REJECTED_CLUSTER_VECTOR = vector
        return None, elapsed_s
    _GATE4_REJECTED_CLUSTER_VECTOR = vector
    if (
        now_s - _GATE4_REJECTED_CLUSTER_STARTED_S
        < GATE4_REANCHOR_PERSISTENCE_S
    ):
        return None, elapsed_s

    _GATE4_REANCHOR_COUNT += 1
    _GATE4_LAST_VECTOR = vector
    _GATE4_ANCHOR_VECTOR = vector
    _GATE4_REJECTED_CLUSTER_STARTED_S = None
    _GATE4_REJECTED_CLUSTER_VECTOR = None
    return vector, elapsed_s


def _yaw_from_quaternion(values: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in values[6:10])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _encode_gate4_thrust(thrust: float) -> float:
    span = (
        GATE4_MAX_THRUST - GATE4_HOVER_THRUST
        if thrust >= GATE4_HOVER_THRUST
        else GATE4_HOVER_THRUST - GATE4_MIN_THRUST
    )
    return tail._clamp((thrust - GATE4_HOVER_THRUST) / span)


def _gate4_safe_handoff(
    values: np.ndarray,
    base_actions: list[float],
    *,
    now_s: float | None = None,
) -> list[float]:
    """Keep the learned Gate-4 tail inside the live-proven safety envelope."""

    global _GATE4_LAST_VISIBLE_YAW_TARGET_RAD, _GATE4_LAST_VISIBLE_YAW_TARGET_S
    control_now_s = time.monotonic() if now_s is None else float(now_s)
    current_yaw_rad = _yaw_from_quaternion(values)
    associated_vector, elapsed_s = _gate4_current_association(
        values,
        control_now_s,
    )
    if associated_vector is None:
        desired_yaw_rad = current_yaw_rad
        if (
            elapsed_s >= GATE4_ACQUISITION_YAW_DELAY_S
            and float(values[10]) > 0.5
        ):
            yaw_error_rad = tail._clamp(float(values[14])) * (math.pi / 4.0)
            yaw_step_rad = tail._clamp(
                yaw_error_rad,
                -GATE4_ACQUISITION_MAX_YAW_STEP_RAD,
                GATE4_ACQUISITION_MAX_YAW_STEP_RAD,
            )
            desired_yaw_rad = _wrap_angle(current_yaw_rad - yaw_step_rad)
            _GATE4_LAST_VISIBLE_YAW_TARGET_RAD = desired_yaw_rad
            _GATE4_LAST_VISIBLE_YAW_TARGET_S = control_now_s
        elif (
            float(values[10]) <= 0.5
            and _GATE4_LAST_VISIBLE_YAW_TARGET_RAD is not None
            and _GATE4_LAST_VISIBLE_YAW_TARGET_S is not None
            and control_now_s - _GATE4_LAST_VISIBLE_YAW_TARGET_S
            <= GATE4_DROPOUT_YAW_CARRY_S
        ):
            carried_delta_rad = tail._clamp(
                _wrap_angle(
                    _GATE4_LAST_VISIBLE_YAW_TARGET_RAD - current_yaw_rad
                ),
                -GATE4_ACQUISITION_MAX_YAW_STEP_RAD,
                GATE4_ACQUISITION_MAX_YAW_STEP_RAD,
            )
            desired_yaw_rad = _wrap_angle(current_yaw_rad + carried_delta_rad)
        return [
            -GATE4_MAX_PITCH_RAD / 0.5,
            0.0,
            _encode_gate4_thrust(GATE4_HOVER_THRUST),
            tail._clamp(desired_yaw_rad / math.pi),
        ]

    forward_m, right_m, down_m = associated_vector
    yaw_error_rad = tail._clamp(float(values[14])) * (math.pi / 4.0)
    aligned = (
        abs(yaw_error_rad) <= GATE4_CENTER_YAW_ERROR_RAD
        and abs(down_m) <= GATE4_CENTER_DOWN_ERROR_M
    )
    if aligned:
        pitch_norm = tail._clamp(
            float(base_actions[0]),
            -GATE4_MAX_PITCH_RAD / 0.5,
            GATE4_ALIGNED_FORWARD_PITCH_RAD / 0.5,
        )
    else:
        pitch_norm = -GATE4_MAX_PITCH_RAD / 0.5

    roll_rad = tail._clamp(
        -GATE4_LATERAL_ROLL_KP * right_m,
        -GATE4_MAX_LATERAL_ROLL_RAD,
        GATE4_MAX_LATERAL_ROLL_RAD,
    )
    if elapsed_s < GATE4_VERTICAL_CONTROL_DELAY_S:
        thrust = GATE4_HOVER_THRUST
    elif forward_m <= GATE4_REANCHOR_MAX_FORWARD_M and down_m > GATE4_CENTER_DOWN_ERROR_M:
        thrust = GATE4_DESCENT_THRUST
    else:
        thrust = tail._clamp(
            GATE4_HOVER_THRUST - GATE4_VERTICAL_THRUST_KP * down_m,
            GATE4_DESCENT_THRUST,
            GATE4_MAX_THRUST,
        )
    desired_yaw_rad = _wrap_angle(current_yaw_rad - yaw_error_rad)
    _GATE4_LAST_VISIBLE_YAW_TARGET_RAD = desired_yaw_rad
    _GATE4_LAST_VISIBLE_YAW_TARGET_S = control_now_s
    return [
        pitch_norm,
        tail._clamp(roll_rad / 0.5),
        _encode_gate4_thrust(thrust),
        tail._clamp(desired_yaw_rad / math.pi),
    ]


def reset() -> None:
    _resolve_prefix().reset_state()
    tail.reset()
    _clear_gate2_state()
    _clear_gate3_state()
    _clear_gate4_state()


def infer(observation: Sequence[float]) -> list[float]:
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    gate = tail._gate_index(values)
    if gate != 1:
        _clear_gate2_state()
    if gate != 2:
        _clear_gate3_state()
    if gate <= 2:
        _clear_gate4_state()
        actions = _resolve_prefix().infer(_legacy_prefix_observation(values))
        if gate == 0:
            actions = _gate1_late_residual_roll_floor(values, actions)
        if gate == 1:
            actions = _gate2_projected_aperture_governor(values, actions)
            actions = _gate2_late_residual_roll_floor(values, actions)
            actions = _gate2_projected_vertical_floor(values, actions)
        if gate == 2:
            actions = tail._promoted_gate3_intercept(values, actions)
            actions = _gate3_close_severe_boost(values, actions)
        return [tail._clamp(value) for value in actions]
    actions = tail.infer(_tail_observation(values))
    if gate == 3:
        return _gate4_safe_handoff(values, actions)
    _clear_gate4_state()
    return actions
