#!/usr/bin/env python3
"""N225: N224 plus a failure-specific Gate-3 vertical thrust guard.

N224 passed official Gates 1 and 2, then commanded strong sub-hover thrust
while its observable Gate-3 trajectory already projected above the vehicle at
the terminal plane.  This wrapper floors only that contradictory action to
hover.  It preserves every other action, uses no privileged state, and latches
only for the current Gate-3 phase.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_n220_observable_tail as n224
import policy_callable_six_gate_composite as tail


EXPECTED_N224_SOURCE_SHA256 = (
    "dda536d07b2548b50ca3d433a222539898d9e7c147f9d29fc054be1a12cbd746"
)
_N224_PATH = Path(n224.__file__).resolve()
if hashlib.sha256(_N224_PATH.read_bytes()).hexdigest() != EXPECTED_N224_SOURCE_SHA256:
    raise RuntimeError("N224 policy source hash mismatch")


GATE3_INDEX = 2
MAX_FORWARD_M = 12.0
MIN_CLOSING_M_S = 1.0
TERMINAL_FORWARD_M = 2.0
PROJECTED_DOWN_TRIGGER_M = 0.0
BASE_THRUST_MAX_NORM = -0.20
THRUST_FLOOR_NORM = 0.0


class Gate3VerticalThrustGuard:
    """Reject strong descent thrust after the visible path projects low."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.activation_count = 0
        self.active_samples = 0
        self.changed_samples = 0
        self.last_projected_down_m: float | None = None

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

        if tail._gate_index(values) != GATE3_INDEX:
            self.active = False
            self.last_projected_down_m = None
            return actions
        if self.active:
            self.active_samples += 1
            governed = max(actions[2], THRUST_FLOOR_NORM)
            if governed != actions[2]:
                actions[2] = governed
                self.changed_samples += 1
            return actions
        if float(values[10]) < 0.5:
            return actions

        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        if not (
            0.0 < forward_m <= MAX_FORWARD_M
            and closing_m_s >= MIN_CLOSING_M_S
            and actions[2] <= BASE_THRUST_MAX_NORM
        ):
            return actions
        down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
        down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
        horizon_s = max(forward_m - TERMINAL_FORWARD_M, 0.0) / closing_m_s
        projected_down_m = down_m + down_rate_m_s * horizon_s
        self.last_projected_down_m = projected_down_m
        if projected_down_m >= PROJECTED_DOWN_TRIGGER_M:
            return actions

        self.active = True
        self.activation_count += 1
        self.active_samples += 1
        governed = max(actions[2], THRUST_FLOOR_NORM)
        if governed != actions[2]:
            actions[2] = governed
            self.changed_samples += 1
        return actions

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "activation_count": self.activation_count,
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "last_projected_down_m": self.last_projected_down_m,
            "parameters": {
                "gate_index": GATE3_INDEX,
                "max_forward_m": MAX_FORWARD_M,
                "minimum_closing_m_s": MIN_CLOSING_M_S,
                "terminal_forward_m": TERMINAL_FORWARD_M,
                "projected_down_trigger_m": PROJECTED_DOWN_TRIGGER_M,
                "base_thrust_max_norm": BASE_THRUST_MAX_NORM,
                "thrust_floor_norm": THRUST_FLOOR_NORM,
                "preserve_pitch_roll_yaw": True,
                "latch_until_official_transition": True,
                "runtime_privileged_state": False,
            },
        }


_CONTROLLER = Gate3VerticalThrustGuard()


def reset() -> None:
    n224.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    return _CONTROLLER.apply(observation, n224.infer(observation))


def controller_snapshot() -> dict:
    return {
        "n225_gate3_vertical_guard": _CONTROLLER.snapshot(),
        "n224": n224.controller_snapshot(),
    }
