#!/usr/bin/env python3
"""N201: exact N200 with a close Gate-1 high-down thrust ceiling."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate3_restored_terminal_level_n199 as n200
import policy_callable_six_gate_composite as tail


EXPECTED_N200_SHA256 = (
    "acf17a600e25e2c4c43e928052789c03dfa934e865e7bbd7e7d39ddd9a056885"
)
_N200_PATH = Path(n200.__file__).resolve()
if hashlib.sha256(_N200_PATH.read_bytes()).hexdigest() != EXPECTED_N200_SHA256:
    raise RuntimeError("N200 policy source hash mismatch")


MAX_FORWARD_M = 6.0
MINIMUM_CLOSING_M_S = 1.0
DOWN_TRIGGER_M = 0.35
THRUST_CEILING_NORM = -0.70


class Gate1HighDownThrustCeiling:
    """Lower thrust only for a close, visible Gate 1 well below the vehicle."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.active_samples = 0
        self.changed_samples = 0
        self.last_forward_m: float | None = None
        self.last_down_m: float | None = None

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "last_forward_m": self.last_forward_m,
            "last_down_m": self.last_down_m,
            "parameters": {
                "max_forward_m": MAX_FORWARD_M,
                "minimum_closing_m_s": MINIMUM_CLOSING_M_S,
                "down_trigger_m": DOWN_TRIGGER_M,
                "thrust_ceiling_norm": THRUST_CEILING_NORM,
                "stateless": True,
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
        self.active = False
        self.last_forward_m = None
        self.last_down_m = None
        if tail._gate_index(values) != 0 or float(values[10]) < 0.5:
            return actions

        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
        self.last_forward_m = forward_m
        self.last_down_m = down_m
        if not (
            0.0 < forward_m <= MAX_FORWARD_M
            and closing_m_s >= MINIMUM_CLOSING_M_S
            and down_m >= DOWN_TRIGGER_M
        ):
            return actions

        self.active = True
        self.active_samples += 1
        governed = min(actions[2], THRUST_CEILING_NORM)
        if governed != actions[2]:
            actions[2] = governed
            self.changed_samples += 1
        return actions


_CONTROLLER = Gate1HighDownThrustCeiling()


def reset() -> None:
    n200.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    return _CONTROLLER.apply(observation, n200.infer(observation))


def controller_snapshot() -> dict:
    return {
        "gate1_high_down_thrust_ceiling": _CONTROLLER.snapshot(),
        "n200": n200.controller_snapshot(),
    }
