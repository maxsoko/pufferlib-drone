#!/usr/bin/env python3
"""N227: N226 plus coordinate-free Gate-4 terminal-plane control.

The N226 flight passed the proven prefix, then the fixed-course N220 tail
initially rolled away from a Gate 4 that was already visible to the right and
oscillated vertical thrust.  This wrapper uses only the deployed relative gate
vector and its observable rate to aim roll and thrust at the terminal plane.
Pitch and yaw remain policy-owned, and every phase except Gate 4 is invariant.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_n225_gate3_vertical_guard as n226
import policy_callable_six_gate_composite as tail


EXPECTED_N226_SOURCE_SHA256 = (
    "c2f104c1264f31df1d22a167dcfb32d47c01b1479229925076ba780e057eb29e"
)
_N226_PATH = Path(n226.__file__).resolve()
if hashlib.sha256(_N226_PATH.read_bytes()).hexdigest() != EXPECTED_N226_SOURCE_SHA256:
    raise RuntimeError("N226 policy source hash mismatch")


GATE4_INDEX = 3
MIN_CLOSING_M_S = 4.1875
MAX_HORIZON_S = 8.0
ROLL_TERMINAL_KP = 0.25
THRUST_TERMINAL_KP = 0.10


def _terminal_error(position_m: float, rate_m_s: float, horizon_s: float) -> float:
    return float(position_m) + float(rate_m_s) * float(horizon_s)


class Gate4TerminalPlaneController:
    """Aim lateral and vertical errors at zero at the observed gate plane."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active_samples = 0
        self.changed_samples = 0
        self.last_terminal_right_m: float | None = None
        self.last_terminal_down_m: float | None = None

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

        if tail._gate_index(values) != GATE4_INDEX or float(values[10]) < 0.5:
            self.last_terminal_right_m = None
            self.last_terminal_down_m = None
            return actions

        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        if forward_m <= 0.0:
            return actions
        forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
        closing_m_s = max(-forward_rate_m_s, MIN_CLOSING_M_S)
        horizon_s = min(forward_m / closing_m_s, MAX_HORIZON_S)

        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
        down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
        terminal_right_m = _terminal_error(right_m, right_rate_m_s, horizon_s)
        terminal_down_m = _terminal_error(down_m, down_rate_m_s, horizon_s)

        governed_roll = tail._clamp(-ROLL_TERMINAL_KP * terminal_right_m)
        governed_thrust = tail._clamp(-THRUST_TERMINAL_KP * terminal_down_m)
        if governed_roll != actions[1] or governed_thrust != actions[2]:
            self.changed_samples += 1
        actions[1] = governed_roll
        actions[2] = governed_thrust
        self.active_samples += 1
        self.last_terminal_right_m = terminal_right_m
        self.last_terminal_down_m = terminal_down_m
        return actions

    def snapshot(self) -> dict:
        return {
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "last_terminal_right_m": self.last_terminal_right_m,
            "last_terminal_down_m": self.last_terminal_down_m,
            "parameters": {
                "gate_index": GATE4_INDEX,
                "minimum_closing_m_s": MIN_CLOSING_M_S,
                "maximum_horizon_s": MAX_HORIZON_S,
                "roll_terminal_kp": ROLL_TERMINAL_KP,
                "thrust_terminal_kp": THRUST_TERMINAL_KP,
                "policy_owned_actions": ["pitch", "yaw"],
                "runtime_privileged_state": False,
            },
        }


_CONTROLLER = Gate4TerminalPlaneController()


def reset() -> None:
    n226.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    return _CONTROLLER.apply(observation, n226.infer(observation))


def controller_snapshot() -> dict:
    return {
        "n227_gate4_terminal_pd": _CONTROLLER.snapshot(),
        "n226": n226.controller_snapshot(),
    }
