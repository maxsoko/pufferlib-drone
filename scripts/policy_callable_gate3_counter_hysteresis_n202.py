#!/usr/bin/env python3
"""N203: exact N202 plus a parameterized Gate-3 counter-bank hysteresis.

The official policy's promoted Gate-3 lateral law changes directly between a
positive intercept bank and a full negative counter-bank at one projected
lateral threshold.  The Windows batch optimizer selects one of three frozen
entry thresholds through ``PUFFER_GATE3_COUNTER_ENTRY_M``.  Once counter-bank
is selected, this wrapper keeps it selected until the projection has moved a
fixed 0.8 m back toward the left.  It retires before the existing close-range
controller at 4 m, so the source-pinned terminal and recovery laws remain in
charge at the aperture.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate3_tight_terminal_level_n201 as n202
import policy_callable_six_gate_composite as tail


EXPECTED_N202_SHA256 = (
    "892f38e34da75f29d7b20a42638ff26a19773518ecab751e4f408842733538dd"
)
_N202_PATH = Path(n202.__file__).resolve()
if hashlib.sha256(_N202_PATH.read_bytes()).hexdigest() != EXPECTED_N202_SHA256:
    raise RuntimeError("N202 policy source hash mismatch")


ENTRY_ENV = "PUFFER_GATE3_COUNTER_ENTRY_M"
ALLOWED_ENTRY_M = (-0.8, -1.2, -1.6)
RELEASE_GAP_M = 0.8
MIN_FORWARD_M_EXCLUSIVE = 4.0
MAX_FORWARD_M = 15.0
TERMINAL_PLANE_FORWARD_M = 2.0
MINIMUM_CLOSING_M_S = 1.0
COUNTER_ROLL_NORM = -1.0
DISTANCE_EPSILON_M = 1e-5


def _entry_threshold_m() -> float:
    raw = os.getenv(ENTRY_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{ENTRY_ENV} is required")
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{ENTRY_ENV} must be numeric") from exc
    if not any(abs(value - allowed) <= 1e-9 for allowed in ALLOWED_ENTRY_M):
        allowed = ", ".join(str(item) for item in ALLOWED_ENTRY_M)
        raise RuntimeError(f"{ENTRY_ENV} must be one of: {allowed}")
    return value


class Gate3CounterHysteresis:
    """Hold the early counter-bank across projected-position chatter."""

    def __init__(self) -> None:
        self.entry_threshold_m = _entry_threshold_m()
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.activation_count = 0
        self.release_count = 0
        self.active_samples = 0
        self.changed_samples = 0
        self.last_projected_right_m: float | None = None

    @property
    def release_threshold_m(self) -> float:
        return self.entry_threshold_m - RELEASE_GAP_M

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "activation_count": self.activation_count,
            "release_count": self.release_count,
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "last_projected_right_m": self.last_projected_right_m,
            "parameters": {
                "entry_threshold_m": self.entry_threshold_m,
                "release_threshold_m": self.release_threshold_m,
                "release_gap_m": RELEASE_GAP_M,
                "min_forward_m_exclusive": MIN_FORWARD_M_EXCLUSIVE,
                "max_forward_m": MAX_FORWARD_M,
                "terminal_plane_forward_m": TERMINAL_PLANE_FORWARD_M,
                "minimum_closing_m_s": MINIMUM_CLOSING_M_S,
                "counter_roll_norm": COUNTER_ROLL_NORM,
                "preserve_pitch_thrust_yaw": True,
            },
        }

    def _clear_active(self) -> None:
        self.active = False
        self.last_projected_right_m = None

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
            self._clear_active()
            return actions
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        if not (
            MIN_FORWARD_M_EXCLUSIVE + DISTANCE_EPSILON_M
            < forward_m
            <= MAX_FORWARD_M + DISTANCE_EPSILON_M
            and closing_m_s >= MINIMUM_CLOSING_M_S
        ):
            self._clear_active()
            return actions

        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        horizon_s = max(
            forward_m - TERMINAL_PLANE_FORWARD_M,
            0.0,
        ) / closing_m_s
        projected_right_m = right_m + right_rate_m_s * horizon_s
        self.last_projected_right_m = projected_right_m

        if self.active:
            if projected_right_m <= self.release_threshold_m:
                self.active = False
                self.release_count += 1
        elif projected_right_m >= self.entry_threshold_m:
            self.active = True
            self.activation_count += 1

        if not self.active:
            return actions
        roll_before = actions[1]
        actions[1] = COUNTER_ROLL_NORM
        self.active_samples += 1
        if actions[1] != roll_before:
            self.changed_samples += 1
        return actions


_CONTROLLER: Gate3CounterHysteresis | None = None


def _controller() -> Gate3CounterHysteresis:
    global _CONTROLLER
    threshold = _entry_threshold_m()
    if _CONTROLLER is None or _CONTROLLER.entry_threshold_m != threshold:
        _CONTROLLER = Gate3CounterHysteresis()
    return _CONTROLLER


def reset() -> None:
    n202.reset()
    _controller().reset()


def infer(observation: Sequence[float]) -> list[float]:
    return _controller().apply(observation, n202.infer(observation))


def controller_snapshot() -> dict:
    return {
        "gate3_counter_hysteresis": _controller().snapshot(),
        "n202": n202.controller_snapshot(),
    }
