#!/usr/bin/env python3
"""N194: exact N193 plus a frozen near-plane Gate-3 recovery scan."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate2_rapid_vertical_deficit_n192 as n193
import policy_callable_six_gate_composite as tail
import policy_callable_six_gate_hybrid as base


EXPECTED_N193_SHA256 = (
    "9da6cde4ed3c5a25f2c16872427ab6aab15ed57e379399ab725fee28b34869f2"
)
_N193_PATH = Path(n193.__file__).resolve()
if hashlib.sha256(_N193_PATH.read_bytes()).hexdigest() != EXPECTED_N193_SHA256:
    raise RuntimeError("N193 policy source hash mismatch")


FEATURE_INDICES = (0, 1, 2, 11, 12, 13, 14)
MAX_FORWARD_M = 3.0
MINIMUM_CLOSING_M_S = 1.0
FROZEN_TRIGGER_SAMPLES = 5
FEATURE_EQUALITY_ATOL = 1.0e-7
LEVEL_ROLL_NORM = 0.0
YAW_SCAN_STEP_RAD = 0.25


class Gate3FrozenObservationRecovery:
    """Stop stale lateral banking and scan until gate motion becomes fresh."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._clear_watchdog()
        self.activation_count = 0
        self.active_samples = 0
        self.changed_samples = 0

    def _clear_watchdog(self) -> None:
        self.active = False
        self.frozen_samples = 0
        self.last_features: np.ndarray | None = None

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "activation_count": self.activation_count,
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "frozen_samples": self.frozen_samples,
            "parameters": {
                "feature_indices": FEATURE_INDICES,
                "max_forward_m": MAX_FORWARD_M,
                "minimum_closing_m_s": MINIMUM_CLOSING_M_S,
                "frozen_trigger_samples": FROZEN_TRIGGER_SAMPLES,
                "feature_equality_atol": FEATURE_EQUALITY_ATOL,
                "level_roll_norm": LEVEL_ROLL_NORM,
                "yaw_scan_step_rad": YAW_SCAN_STEP_RAD,
                "preserve_pitch_thrust": True,
                "release_on_fresh_motion": True,
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
        if tail._gate_index(values) != 2 or float(values[10]) < 0.5:
            self._clear_watchdog()
            return actions
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        if not (
            0.0 < forward_m <= MAX_FORWARD_M
            and closing_m_s >= MINIMUM_CLOSING_M_S
        ):
            self._clear_watchdog()
            return actions
        features = values[list(FEATURE_INDICES)]
        unchanged = bool(
            self.last_features is not None
            and np.max(np.abs(features - self.last_features))
            <= FEATURE_EQUALITY_ATOL
        )
        self.last_features = features.copy()
        if unchanged:
            self.frozen_samples += 1
        else:
            self.active = False
            self.frozen_samples = 1
        if not self.active and self.frozen_samples >= FROZEN_TRIGGER_SAMPLES:
            self.active = True
            self.activation_count += 1
        if not self.active:
            return actions

        roll_before = actions[1]
        yaw_before = actions[3]
        yaw_error_rad = float(values[14]) * (math.pi / 4.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        bearing_sign = yaw_error_rad if abs(yaw_error_rad) >= 0.05 else right_m
        direction = 1.0 if bearing_sign >= 0.0 else -1.0
        current_yaw_rad = base._yaw_from_quaternion(values)
        desired_yaw_rad = base._wrap_angle(
            current_yaw_rad - direction * YAW_SCAN_STEP_RAD
        )
        actions[1] = LEVEL_ROLL_NORM
        actions[3] = tail._clamp(desired_yaw_rad / math.pi)
        self.active_samples += 1
        if actions[1] != roll_before or actions[3] != yaw_before:
            self.changed_samples += 1
        return actions


_CONTROLLER = Gate3FrozenObservationRecovery()


def reset() -> None:
    n193.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    actions = n193.infer(observation)
    return _CONTROLLER.apply(observation, actions)


def controller_snapshot() -> dict:
    return {
        "gate3_frozen_observation_recovery": _CONTROLLER.snapshot(),
        "n193": n193.controller_snapshot(),
    }
