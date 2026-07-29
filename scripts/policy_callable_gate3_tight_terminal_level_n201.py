#!/usr/bin/env python3
"""N202: N199 plus N201 Gate-1 ceiling and tighter restored Gate-3 leveling."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate1_high_down_thrust_ceiling_n200 as n201_module
import policy_callable_gate3_later_terminal_level_n195 as n196_module
import policy_callable_six_gate_composite as tail


EXPECTED_N201_SHA256 = (
    "1c555692497ebb2e0ec35c309158ee55813d5caad299709483439bf636c4c1e9"
)
EXPECTED_N196_SHA256 = (
    "3e2fb526ae0115b36c4dfbdfe1c5a5bb5a6265caabd3a45a3c9ea8ed1077c2df"
)


def _assert_source_hash(module, expected: str, name: str) -> None:
    path = Path(module.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{name} policy source hash mismatch")


_assert_source_hash(n201_module, EXPECTED_N201_SHA256, "N201")
_assert_source_hash(n196_module, EXPECTED_N196_SHA256, "N196 terminal controller")


MAX_FORWARD_M = n196_module.MAX_FORWARD_M
MINIMUM_CLOSING_M_S = n196_module.MINIMUM_CLOSING_M_S
TERMINAL_PLANE_FORWARD_M = n196_module.TERMINAL_PLANE_FORWARD_M
MAX_ABS_PROJECTED_RIGHT_M = 0.30
MAX_ABS_PROJECTED_DOWN_M = n196_module.MAX_ABS_PROJECTED_DOWN_M
LEVEL_ROLL_NORM = n196_module.LEVEL_ROLL_NORM


class Gate3TightTerminalSafeLevel:
    """Level Gate-3 roll only inside a tighter projected lateral corridor."""

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


_N199 = n201_module.n200.n199
_GATE1 = n201_module.Gate1HighDownThrustCeiling()
_TIGHT_TERMINAL = Gate3TightTerminalSafeLevel()


def reset() -> None:
    _N199.reset()
    _GATE1.reset()
    _TIGHT_TERMINAL.reset()


def infer(observation: Sequence[float]) -> list[float]:
    actions = _N199.infer(observation)
    actions = _TIGHT_TERMINAL.apply(observation, actions)
    return _GATE1.apply(observation, actions)


def controller_snapshot() -> dict:
    return {
        "gate1_high_down_thrust_ceiling": _GATE1.snapshot(),
        "gate3_tight_terminal_safe_level": _TIGHT_TERMINAL.snapshot(),
        "n199": _N199.controller_snapshot(),
    }
