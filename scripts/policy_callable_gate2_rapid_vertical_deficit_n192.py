#!/usr/bin/env python3
"""N193: exact N192 plus a latched early rapid Gate-2 vertical correction."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate3_terminal_level_n191 as n192
import policy_callable_six_gate_composite as tail


EXPECTED_N192_SHA256 = (
    "3750e1620f4a7d244b93c370f31ec135a89572e64a60888d7ea032c244949fae"
)
_N192_PATH = Path(n192.__file__).resolve()
if hashlib.sha256(_N192_PATH.read_bytes()).hexdigest() != EXPECTED_N192_SHA256:
    raise RuntimeError("N192 policy source hash mismatch")


MIN_FORWARD_M = 3.0
MAX_FORWARD_M = 7.0
DOWN_RATE_TRIGGER_M_S = -2.9
PROJECTED_DOWN_TRIGGER_M = -1.0
BASE_THRUST_MAX_NORM = 0.05
PROJECTED_DOWN_RELEASE_M = -0.4
THRUST_FLOOR_NORM = 1.0
INTERCEPT_SPEED_FLOOR_M_S = 3.0


class Gate2RapidVerticalDeficit:
    """Correct a fast vertical deficit while the inherited thrust is dormant."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.activation_count = 0
        self.active_samples = 0
        self.changed_samples = 0
        self.last_projected_down_m: float | None = None

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "activation_count": self.activation_count,
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "last_projected_down_m": self.last_projected_down_m,
            "parameters": {
                "min_forward_m": MIN_FORWARD_M,
                "max_forward_m": MAX_FORWARD_M,
                "down_rate_trigger_m_s": DOWN_RATE_TRIGGER_M_S,
                "projected_down_trigger_m": PROJECTED_DOWN_TRIGGER_M,
                "base_thrust_max_norm": BASE_THRUST_MAX_NORM,
                "projected_down_release_m": PROJECTED_DOWN_RELEASE_M,
                "thrust_floor_norm": THRUST_FLOOR_NORM,
                "intercept_speed_floor_m_s": INTERCEPT_SPEED_FLOOR_M_S,
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
        if tail._gate_index(values) != 1 or float(values[10]) < 0.5:
            self.active = False
            self.last_projected_down_m = None
            return actions
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
        down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
        horizon_s = forward_m / max(closing_m_s, INTERCEPT_SPEED_FLOOR_M_S)
        projected_down_m = down_m + down_rate_m_s * horizon_s
        self.last_projected_down_m = projected_down_m
        if self.active and projected_down_m >= PROJECTED_DOWN_RELEASE_M:
            self.active = False
        if (
            not self.active
            and MIN_FORWARD_M <= forward_m <= MAX_FORWARD_M
            and closing_m_s >= 1.0
            and down_rate_m_s < DOWN_RATE_TRIGGER_M_S
            and projected_down_m < PROJECTED_DOWN_TRIGGER_M
            and actions[2] <= BASE_THRUST_MAX_NORM
        ):
            self.active = True
            self.activation_count += 1
        if not self.active:
            return actions
        self.active_samples += 1
        governed = max(actions[2], THRUST_FLOOR_NORM)
        if governed != actions[2]:
            actions[2] = governed
            self.changed_samples += 1
        return actions


_CONTROLLER = Gate2RapidVerticalDeficit()


def reset() -> None:
    n192.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    actions = n192.infer(observation)
    return _CONTROLLER.apply(observation, actions)


def controller_snapshot() -> dict:
    return {
        "gate2_rapid_vertical_deficit": _CONTROLLER.snapshot(),
        "n192": n192.controller_snapshot(),
    }
