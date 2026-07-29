#!/usr/bin/env python3
"""N204: exact H12/N203 plus corrected visible Gate-4 yaw geometry."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate3_counter_hysteresis_n202 as n203
import policy_callable_six_gate_composite as tail


EXPECTED_N203_SHA256 = (
    "b357536767a62beb85ba707a297b67ef5191998a425f6e0912a9882e07ee5cbe"
)
_N203_PATH = Path(n203.__file__).resolve()
if hashlib.sha256(_N203_PATH.read_bytes()).hexdigest() != EXPECTED_N203_SHA256:
    raise RuntimeError("N203 policy source hash mismatch")


MAX_YAW_STEP_RAD = 0.35


def _yaw_from_quaternion(values: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in values[6:10])
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


class Gate4VisibleYawSign:
    """Turn toward the signed camera bearing while official Gate 4 is active."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active_samples = 0
        self.changed_samples = 0
        self.last_yaw_error_rad: float | None = None
        self.last_desired_yaw_rad: float | None = None

    def snapshot(self) -> dict:
        return {
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "last_yaw_error_rad": self.last_yaw_error_rad,
            "last_desired_yaw_rad": self.last_desired_yaw_rad,
            "parameters": {
                "gate_index": 3,
                "max_yaw_step_rad": MAX_YAW_STEP_RAD,
                "desired_yaw_law": "current_yaw_plus_camera_yaw_error",
                "preserve_pitch_roll_thrust": True,
                "dropout_passthrough": True,
            },
        }

    def apply(
        self,
        observation: Sequence[float],
        base_actions: Sequence[float],
    ) -> list[float]:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (32,):
            raise ValueError(f"expected observation shape (32,), got {values.shape}")
        if len(base_actions) != 4:
            raise ValueError("expected four base actions")
        actions = [tail._clamp(float(value)) for value in base_actions]
        if tail._gate_index(values) != 3 or float(values[10]) <= 0.5:
            return actions

        yaw_error_rad = tail._clamp(float(values[14])) * (math.pi / 4.0)
        yaw_step_rad = tail._clamp(
            yaw_error_rad,
            -MAX_YAW_STEP_RAD,
            MAX_YAW_STEP_RAD,
        )
        current_yaw_rad = _yaw_from_quaternion(values)
        desired_yaw_rad = _wrap_angle(current_yaw_rad + yaw_step_rad)
        yaw_before = actions[3]
        actions[3] = tail._clamp(desired_yaw_rad / math.pi)
        self.active_samples += 1
        if actions[3] != yaw_before:
            self.changed_samples += 1
        self.last_yaw_error_rad = yaw_error_rad
        self.last_desired_yaw_rad = desired_yaw_rad
        return actions


_CONTROLLER = Gate4VisibleYawSign()


def reset() -> None:
    n203.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    return _CONTROLLER.apply(observation, n203.infer(observation))


def controller_snapshot() -> dict:
    return {
        "gate4_visible_yaw_sign": _CONTROLLER.snapshot(),
        "n203": n203.controller_snapshot(),
    }
