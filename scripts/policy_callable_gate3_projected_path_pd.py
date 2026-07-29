#!/usr/bin/env python3
"""N184 policy with one bounded, smooth Gate-3-only roll controller.

The controller anchors a cubic path at the first visible official Gate-3
sample inside 12 m.  It replaces normalized roll only; the exact N184 policy
continues to own pitch, thrust, yaw, every earlier gate, and the six-gate tail.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_six_gate_hybrid as base
import policy_callable_six_gate_composite as tail


EXPECTED_BASE_SHA256 = (
    "134337cc8f91b6869edc921e90f55ce4d06c5664591d50a5723d7f14af544514"
)
_BASE_PATH = Path(base.__file__).resolve()
if hashlib.sha256(_BASE_PATH.read_bytes()).hexdigest() != EXPECTED_BASE_SHA256:
    raise RuntimeError("N184 base policy source hash mismatch")


@dataclasses.dataclass(frozen=True)
class Parameters:
    admission_forward_m: float = 12.0
    minimum_closing_m_s: float = 1.0
    path_power: float = 3.0
    terminal_right_m: float = -0.10
    position_gain: float = 0.20
    rate_gain: float = 1.0
    max_roll_norm: float = 1.0
    max_roll_slew_norm_s: float = 4.0
    nominal_control_hz: float = 60.0


PARAMETERS = Parameters()


class Gate3ProjectedPathPD:
    def __init__(self, parameters: Parameters = PARAMETERS) -> None:
        self.parameters = parameters
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.initial_forward_m = 0.0
        self.initial_right_m = 0.0
        self.previous_roll_norm = 0.0
        self.steps_active = 0
        self.last_requested_roll_norm = 0.0
        self.last_slew_norm_s = 0.0

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "initial_forward_m": self.initial_forward_m,
            "initial_right_m": self.initial_right_m,
            "previous_roll_norm": self.previous_roll_norm,
            "steps_active": self.steps_active,
            "last_requested_roll_norm": self.last_requested_roll_norm,
            "last_slew_norm_s": self.last_slew_norm_s,
            "parameters": dataclasses.asdict(self.parameters),
        }

    def apply(
        self,
        observation: Sequence[float],
        base_actions: Sequence[float],
        *,
        dt_s: float | None = None,
    ) -> list[float]:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (32,):
            raise ValueError(f"expected observation shape (32,), got {values.shape}")
        if len(base_actions) != 4:
            raise ValueError("expected four base actions")
        actions = [tail._clamp(float(value)) for value in base_actions]
        gate = tail._gate_index(values)
        if gate != 2:
            self.reset()
            return actions

        visible = float(values[10]) >= 0.5
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        forward_rate_m_s = tail._inverse_tanh_norm(float(values[0]), 5.0)
        right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
        closing_m_s = -forward_rate_m_s
        p = self.parameters

        if not self.active:
            if not (
                visible
                and 0.0 < forward_m <= p.admission_forward_m
                and closing_m_s >= p.minimum_closing_m_s
            ):
                return actions
            self.active = True
            self.initial_forward_m = max(forward_m, 0.25)
            self.initial_right_m = right_m
            self.previous_roll_norm = actions[1]

        if not visible or forward_m <= 0.0:
            actions[1] = self.previous_roll_norm
            return actions

        ratio = tail._clamp(forward_m / self.initial_forward_m, 0.0, 1.0)
        desired_right_m = p.terminal_right_m + (
            self.initial_right_m - p.terminal_right_m
        ) * ratio**p.path_power
        desired_right_rate_m_s = -(
            self.initial_right_m - p.terminal_right_m
        ) * p.path_power / self.initial_forward_m * (
            ratio ** max(p.path_power - 1.0, 0.0)
        ) * max(closing_m_s, 0.0)
        requested_roll = (
            p.position_gain * (desired_right_m - right_m)
            + p.rate_gain * (desired_right_rate_m_s - right_rate_m_s)
        )
        requested_roll = tail._clamp(
            requested_roll, -p.max_roll_norm, p.max_roll_norm
        )
        self.last_requested_roll_norm = requested_roll

        step_dt_s = 1.0 / p.nominal_control_hz if dt_s is None else float(dt_s)
        if not math.isfinite(step_dt_s) or step_dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        step_dt_s = min(step_dt_s, 0.25)
        max_delta = p.max_roll_slew_norm_s * step_dt_s
        governed_roll = tail._clamp(
            requested_roll,
            self.previous_roll_norm - max_delta,
            self.previous_roll_norm + max_delta,
        )
        self.last_slew_norm_s = abs(
            governed_roll - self.previous_roll_norm
        ) / step_dt_s
        governed_roll = tail._clamp(
            governed_roll, -p.max_roll_norm, p.max_roll_norm
        )
        actions[1] = governed_roll
        self.previous_roll_norm = governed_roll
        self.steps_active += 1
        return actions


_CONTROLLER = Gate3ProjectedPathPD()


def reset() -> None:
    base.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    actions = base.infer(observation)
    return _CONTROLLER.apply(observation, actions)


def controller_snapshot() -> dict:
    return _CONTROLLER.snapshot()
