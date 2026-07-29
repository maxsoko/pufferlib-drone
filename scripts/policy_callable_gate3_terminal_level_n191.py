#!/usr/bin/env python3
"""N192: exact N191 plus terminal Gate-3 roll leveling when projection is safe."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate2_severe_vertical_n190 as n191
import policy_callable_six_gate_composite as tail
import policy_callable_six_gate_hybrid as base


EXPECTED_N191_SHA256 = (
    "f301b682edc10efffc3c131910ea31a479cd323beb888c1d3ddccfd1f13bd73a"
)
EXPECTED_BASE_SHA256 = (
    "134337cc8f91b6869edc921e90f55ce4d06c5664591d50a5723d7f14af544514"
)


def _assert_source_hash(module, expected: str, name: str) -> None:
    path = Path(module.__file__).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{name} policy source hash mismatch")


_assert_source_hash(n191, EXPECTED_N191_SHA256, "N191")
_assert_source_hash(base, EXPECTED_BASE_SHA256, "N184 terminal-level source")


class Gate3TerminalSafeLevel:
    """Reuse N184's aperture projection to remove terminal Gate-3 bank."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        base._clear_gate3_state()
        self.active_samples = 0
        self.activation_count = 0
        self.changed_samples = 0

    def snapshot(self) -> dict:
        return {
            "active": base._GATE3_TERMINAL_LEVEL_LATCHED,
            "active_samples": self.active_samples,
            "activation_count": self.activation_count,
            "changed_samples": self.changed_samples,
            "parameters": {
                "max_forward_m": base.GATE3_TERMINAL_LEVEL_FORWARD_M,
                "max_abs_projected_right_m": (
                    base.GATE3_TERMINAL_MAX_ABS_PROJECTED_RIGHT_M
                ),
                "max_abs_projected_down_m": (
                    base.GATE3_TERMINAL_MAX_ABS_PROJECTED_DOWN_M
                ),
                "terminal_plane_forward_m": 2.0,
                "level_roll_norm": 0.0,
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
            base._clear_gate3_state()
            return actions
        latched_before = base._GATE3_TERMINAL_LEVEL_LATCHED
        roll_before = actions[1]
        actions = base._gate3_terminal_level(values, actions)
        latched_after = base._GATE3_TERMINAL_LEVEL_LATCHED
        if latched_after:
            self.active_samples += 1
        if not latched_before and latched_after:
            self.activation_count += 1
        if actions[1] != roll_before:
            self.changed_samples += 1
        return [tail._clamp(value) for value in actions]


_CONTROLLER = Gate3TerminalSafeLevel()


def reset() -> None:
    n191.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    actions = n191.infer(observation)
    return _CONTROLLER.apply(observation, actions)


def controller_snapshot() -> dict:
    return {
        "gate3_terminal_safe_level": _CONTROLLER.snapshot(),
        "n191": n191.controller_snapshot(),
    }
