#!/usr/bin/env python3
"""N197: exact N196 plus persistent zero-confidence Gate-3 recovery."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate3_later_terminal_level_n195 as n196
import policy_callable_six_gate_composite as tail
import policy_callable_six_gate_hybrid as base


EXPECTED_N196_SHA256 = (
    "3e2fb526ae0115b36c4dfbdfe1c5a5bb5a6265caabd3a45a3c9ea8ed1077c2df"
)
EXPECTED_BASE_SHA256 = (
    "134337cc8f91b6869edc921e90f55ce4d06c5664591d50a5723d7f14af544514"
)


def _assert_source_hash(module, expected: str, name: str) -> None:
    path = Path(module.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{name} policy source hash mismatch")


_assert_source_hash(n196, EXPECTED_N196_SHA256, "N196")
_assert_source_hash(base, EXPECTED_BASE_SHA256, "N184 attitude helpers")


CONFIDENCE_FIELD_INDEX = 30
MAX_FORWARD_M = 3.0
MINIMUM_CLOSING_M_S = 1.0
ZERO_CONFIDENCE_TRIGGER_SAMPLES = 5
LEVEL_ROLL_NORM = 0.0
YAW_SCAN_STEP_RAD = 0.25


class Gate3ZeroConfidenceRecovery:
    """Level and yaw-scan after persistent legal zero detector confidence."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._clear_watchdog()
        self.activation_count = 0
        self.active_samples = 0
        self.changed_samples = 0

    def _clear_watchdog(self) -> None:
        self.active = False
        self.zero_confidence_samples = 0

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "activation_count": self.activation_count,
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "zero_confidence_samples": self.zero_confidence_samples,
            "parameters": {
                "confidence_field_index": CONFIDENCE_FIELD_INDEX,
                "maximum_confidence": 0.0,
                "max_forward_m": MAX_FORWARD_M,
                "minimum_closing_m_s": MINIMUM_CLOSING_M_S,
                "zero_confidence_trigger_samples": ZERO_CONFIDENCE_TRIGGER_SAMPLES,
                "level_roll_norm": LEVEL_ROLL_NORM,
                "yaw_scan_step_rad": YAW_SCAN_STEP_RAD,
                "preserve_pitch_thrust": True,
                "release_on_positive_confidence": True,
                "release_on_ineligible_sample": True,
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
        if float(values[CONFIDENCE_FIELD_INDEX]) > 0.0:
            self._clear_watchdog()
            return actions

        self.zero_confidence_samples += 1
        if (
            not self.active
            and self.zero_confidence_samples >= ZERO_CONFIDENCE_TRIGGER_SAMPLES
        ):
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


_RECOVERY = Gate3ZeroConfidenceRecovery()


def reset() -> None:
    n196.reset()
    _RECOVERY.reset()


def infer(observation: Sequence[float]) -> list[float]:
    return _RECOVERY.apply(observation, n196.infer(observation))


def controller_snapshot() -> dict:
    return {
        "gate3_zero_confidence_recovery": _RECOVERY.snapshot(),
        "n196": n196.controller_snapshot(),
    }
