#!/usr/bin/env python3
"""N198: exact N197 composition with final-only Gate-3 roll leveling."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate2_severe_vertical_n190 as n191
import policy_callable_gate2_rapid_vertical_deficit_n192 as n193_module
import policy_callable_gate3_frozen_observation_recovery_n193 as n194_module
import policy_callable_gate2_frozen_observation_dropout_n194 as n195_module
import policy_callable_gate3_zero_confidence_recovery_n196 as n197_module
import policy_callable_six_gate_composite as tail


EXPECTED_SOURCE_SHA256 = {
    "N191": (
        n191,
        "f301b682edc10efffc3c131910ea31a479cd323beb888c1d3ddccfd1f13bd73a",
    ),
    "N193 controller": (
        n193_module,
        "9da6cde4ed3c5a25f2c16872427ab6aab15ed57e379399ab725fee28b34869f2",
    ),
    "N194 controller": (
        n194_module,
        "a68c07a2a55f61fc1b9ad6522a27e3f6b0ba206dab6dd554392f8c5da417a6a7",
    ),
    "N195 sanitizer": (
        n195_module,
        "46a075eafd49a1e44a14e32af458a26e8d411e4cb507b510c6aaaa7ee6b97b4a",
    ),
    "N197 controller": (
        n197_module,
        "ca2ceb12ecbc62a721bf3f34b7cd6b2675592626604899153c3ab7b92754b210",
    ),
}


def _assert_source_hash(module, expected: str, name: str) -> None:
    path = Path(module.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{name} policy source hash mismatch")


for _source_name, (_source_module, _expected_hash) in EXPECTED_SOURCE_SHA256.items():
    _assert_source_hash(_source_module, _expected_hash, _source_name)


MAX_FORWARD_M = 2.0
MINIMUM_CLOSING_M_S = 1.0
TERMINAL_PLANE_FORWARD_M = 2.0
MAX_ABS_PROJECTED_RIGHT_M = 0.5
MAX_ABS_PROJECTED_DOWN_M = 1.0
LEVEL_ROLL_NORM = 0.0


class Gate3FinalTerminalSafeLevel:
    """Level roll only after a safe Gate-3 projection inside 2.0 metres."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.active_samples = 0
        self.activation_count = 0
        self.changed_samples = 0

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "active_samples": self.active_samples,
            "activation_count": self.activation_count,
            "changed_samples": self.changed_samples,
            "parameters": {
                "max_forward_m": MAX_FORWARD_M,
                "minimum_closing_m_s": MINIMUM_CLOSING_M_S,
                "terminal_plane_forward_m": TERMINAL_PLANE_FORWARD_M,
                "max_abs_projected_right_m": MAX_ABS_PROJECTED_RIGHT_M,
                "max_abs_projected_down_m": MAX_ABS_PROJECTED_DOWN_M,
                "level_roll_norm": LEVEL_ROLL_NORM,
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
            self.active = False
            return actions

        was_active = self.active
        if not self.active and float(values[10]) >= 0.5:
            forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
            closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
            if (
                0.0 < forward_m <= MAX_FORWARD_M
                and closing_m_s >= MINIMUM_CLOSING_M_S
            ):
                horizon_s = (
                    max(forward_m - TERMINAL_PLANE_FORWARD_M, 0.0)
                    / closing_m_s
                )
                right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
                down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
                right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
                down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
                projected_right_m = right_m + right_rate_m_s * horizon_s
                projected_down_m = down_m + down_rate_m_s * horizon_s
                self.active = (
                    abs(projected_right_m) <= MAX_ABS_PROJECTED_RIGHT_M
                    and abs(projected_down_m) <= MAX_ABS_PROJECTED_DOWN_M
                )
        if not self.active:
            return actions
        if not was_active:
            self.activation_count += 1
        roll_before = actions[1]
        actions[1] = LEVEL_ROLL_NORM
        self.active_samples += 1
        if actions[1] != roll_before:
            self.changed_samples += 1
        return actions


_SANITIZER = n195_module.Gate2FrozenObservationDropout()
_FINAL_TERMINAL = Gate3FinalTerminalSafeLevel()
_RAPID_VERTICAL = n193_module.Gate2RapidVerticalDeficit()
_FROZEN_RECOVERY = n194_module.Gate3FrozenObservationRecovery()
_ZERO_CONFIDENCE = n197_module.Gate3ZeroConfidenceRecovery()


def reset() -> None:
    n191.reset()
    _SANITIZER.reset()
    _FINAL_TERMINAL.reset()
    _RAPID_VERTICAL.reset()
    _FROZEN_RECOVERY.reset()
    _ZERO_CONFIDENCE.reset()


def infer(observation: Sequence[float]) -> list[float]:
    sanitized = _SANITIZER.sanitize(observation)
    actions = n191.infer(sanitized)
    actions = _FINAL_TERMINAL.apply(sanitized, actions)
    actions = _RAPID_VERTICAL.apply(sanitized, actions)
    actions = _FROZEN_RECOVERY.apply(sanitized, actions)
    return _ZERO_CONFIDENCE.apply(sanitized, actions)


def controller_snapshot() -> dict:
    return {
        "gate2_frozen_observation_dropout": _SANITIZER.snapshot(),
        "gate3_final_terminal_safe_level": _FINAL_TERMINAL.snapshot(),
        "gate2_rapid_vertical_deficit": _RAPID_VERTICAL.snapshot(),
        "gate3_frozen_observation_recovery": _FROZEN_RECOVERY.snapshot(),
        "gate3_zero_confidence_recovery": _ZERO_CONFIDENCE.snapshot(),
        "n191": n191.controller_snapshot(),
    }
