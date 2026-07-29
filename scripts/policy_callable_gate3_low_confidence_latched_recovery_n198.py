#!/usr/bin/env python3
"""N199: exact N198 plus latched weak-confidence Gate-3 recovery."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate3_final_terminal_level_n197 as n198
import policy_callable_six_gate_composite as tail
import policy_callable_six_gate_hybrid as base


EXPECTED_N198_SHA256 = (
    "cfd9308910dbe63a857a67a3b96d53204a2ad929c9ad9b1a93787ba9392e0724"
)
EXPECTED_BASE_SHA256 = (
    "134337cc8f91b6869edc921e90f55ce4d06c5664591d50a5723d7f14af544514"
)


def _assert_source_hash(module, expected: str, name: str) -> None:
    path = Path(module.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{name} policy source hash mismatch")


_assert_source_hash(n198, EXPECTED_N198_SHA256, "N198")
_assert_source_hash(base, EXPECTED_BASE_SHA256, "N184 attitude helpers")


CONFIDENCE_FIELD_INDEX = 30
MAXIMUM_CONFIDENCE = 0.025
MAX_FORWARD_M = 3.0
MINIMUM_CLOSING_M_S = 1.0
LOW_CONFIDENCE_TRIGGER_SAMPLES = 5
LEVEL_ROLL_NORM = 0.0
YAW_SCAN_STEP_RAD = 0.25


class Gate3LowConfidenceLatchedRecovery:
    """Level and yaw-scan after persistent weak confidence until transition."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.low_confidence_samples = 0
        self.activation_count = 0
        self.active_samples = 0
        self.changed_samples = 0

    def _release_for_transition(self) -> None:
        self.active = False
        self.low_confidence_samples = 0

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "activation_count": self.activation_count,
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "low_confidence_samples": self.low_confidence_samples,
            "parameters": {
                "confidence_field_index": CONFIDENCE_FIELD_INDEX,
                "maximum_confidence": MAXIMUM_CONFIDENCE,
                "max_forward_m": MAX_FORWARD_M,
                "minimum_closing_m_s": MINIMUM_CLOSING_M_S,
                "low_confidence_trigger_samples": LOW_CONFIDENCE_TRIGGER_SAMPLES,
                "level_roll_norm": LEVEL_ROLL_NORM,
                "yaw_scan_step_rad": YAW_SCAN_STEP_RAD,
                "preserve_pitch_thrust": True,
                "latch_until_official_transition": True,
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
        if tail._gate_index(values) != 2:
            self._release_for_transition()
            return actions

        if not self.active:
            forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
            closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
            eligible = (
                float(values[10]) >= 0.5
                and 0.0 < forward_m <= MAX_FORWARD_M
                and closing_m_s >= MINIMUM_CLOSING_M_S
                and float(values[CONFIDENCE_FIELD_INDEX]) <= MAXIMUM_CONFIDENCE
            )
            if not eligible:
                self.low_confidence_samples = 0
                return actions
            self.low_confidence_samples += 1
            if self.low_confidence_samples < LOW_CONFIDENCE_TRIGGER_SAMPLES:
                return actions
            self.active = True
            self.activation_count += 1

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


_RECOVERY = Gate3LowConfidenceLatchedRecovery()


def reset() -> None:
    n198.reset()
    _RECOVERY.reset()


def infer(observation: Sequence[float]) -> list[float]:
    return _RECOVERY.apply(observation, n198.infer(observation))


def controller_snapshot() -> dict:
    return {
        "gate3_low_confidence_latched_recovery": _RECOVERY.snapshot(),
        "n198": n198.controller_snapshot(),
    }
