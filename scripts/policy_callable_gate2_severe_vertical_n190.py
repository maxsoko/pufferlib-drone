#!/usr/bin/env python3
"""N191: exact N190 plus an early severe Gate-2 vertical floor."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate2_vertical_margin_n189 as n190
import policy_callable_six_gate_composite as tail


EXPECTED_N190_SHA256 = (
    "70292925b0748e959eb654febf994afd623143b319bb241ec2e93312ef9e24fb"
)
_N190_PATH = Path(n190.__file__).resolve()
if hashlib.sha256(_N190_PATH.read_bytes()).hexdigest() != EXPECTED_N190_SHA256:
    raise RuntimeError("N190 policy source hash mismatch")


MAX_FORWARD_M = 16.0
PROJECTED_DOWN_TRIGGER_M = -4.0
THRUST_FLOOR_NORM = 1.0
INTERCEPT_SPEED_FLOOR_M_S = 3.0


class Gate2SevereVerticalFloor:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.active_samples = 0
        self.changed_samples = 0
        self.last_projected_down_m: float | None = None

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "last_projected_down_m": self.last_projected_down_m,
            "parameters": {
                "max_forward_m": MAX_FORWARD_M,
                "projected_down_trigger_m": PROJECTED_DOWN_TRIGGER_M,
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
        self.active = False
        self.last_projected_down_m = None
        if tail._gate_index(values) != 1 or float(values[10]) < 0.5:
            return actions
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        if not 0.0 < forward_m <= MAX_FORWARD_M:
            return actions
        closing_m_s = -tail._inverse_tanh_norm(float(values[0]), 5.0)
        horizon_s = forward_m / max(closing_m_s, INTERCEPT_SPEED_FLOOR_M_S)
        down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
        down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
        projected_down_m = down_m + down_rate_m_s * horizon_s
        self.last_projected_down_m = projected_down_m
        if projected_down_m >= PROJECTED_DOWN_TRIGGER_M:
            return actions
        self.active = True
        self.active_samples += 1
        governed = max(actions[2], THRUST_FLOOR_NORM)
        if governed != actions[2]:
            self.changed_samples += 1
            actions[2] = governed
        return actions


_CONTROLLER = Gate2SevereVerticalFloor()


def reset() -> None:
    n190.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    actions = n190.infer(observation)
    return _CONTROLLER.apply(observation, actions)


def controller_snapshot() -> dict:
    return {
        "gate2_severe_vertical": _CONTROLLER.snapshot(),
        "n190": n190.controller_snapshot(),
    }
