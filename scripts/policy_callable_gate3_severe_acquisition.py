#!/usr/bin/env python3
"""N184 plus a bounded Gate-3 severe far-offset acquisition guard.

Normal Gate-3 approaches are bit-for-bit delegated to N184. When the current
official Gate 3 is visible 15--35 m ahead but at least 8 m laterally displaced,
the guard temporarily uses the already bounded Gate-4 acquisition primitive:
level roll, hover thrust, conservative pitch, and a limited absolute-yaw step
toward the opening. It hands back to N184 after the opening is within the
coherent acquisition envelope.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_six_gate_composite as tail
import policy_callable_six_gate_hybrid as base


EXPECTED_BASE_SHA256 = (
    "134337cc8f91b6869edc921e90f55ce4d06c5664591d50a5723d7f14af544514"
)
_BASE_PATH = Path(base.__file__).resolve()
if hashlib.sha256(_BASE_PATH.read_bytes()).hexdigest() != EXPECTED_BASE_SHA256:
    raise RuntimeError("N184 base policy source hash mismatch")


@dataclasses.dataclass(frozen=True)
class Parameters:
    trigger_min_forward_m: float = 15.0
    trigger_max_forward_m: float = 35.0
    trigger_min_abs_right_m: float = 8.0
    release_max_abs_right_m: float = 4.5
    release_max_abs_yaw_error_rad: float = 0.12
    acquisition_pitch_norm: float = -0.24
    acquisition_roll_norm: float = 0.0
    acquisition_thrust_norm: float = 0.0
    max_yaw_step_rad: float = 0.35
    max_dropout_ticks: int = 30


PARAMETERS = Parameters()


def _yaw_from_quaternion(values: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in values[6:10])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


class Gate3SevereAcquisition:
    def __init__(self, parameters: Parameters = PARAMETERS) -> None:
        self.parameters = parameters
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.last_yaw_target_rad: float | None = None
        self.dropout_ticks = 0
        self.activations = 0
        self.releases = 0
        self.dropout_deactivations = 0
        self.steps_active = 0

    def _deactivate(self) -> None:
        self.active = False
        self.last_yaw_target_rad = None
        self.dropout_ticks = 0

    def snapshot(self) -> dict:
        return {
            "active": self.active,
            "last_yaw_target_rad": self.last_yaw_target_rad,
            "dropout_ticks": self.dropout_ticks,
            "activations": self.activations,
            "releases": self.releases,
            "dropout_deactivations": self.dropout_deactivations,
            "steps_active": self.steps_active,
            "parameters": dataclasses.asdict(self.parameters),
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
            self._deactivate()
            return actions

        p = self.parameters
        visible = float(values[10]) >= 0.5
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        yaw_error_rad = tail._clamp(float(values[14])) * (math.pi / 4.0)

        if not self.active:
            severe = (
                visible
                and p.trigger_min_forward_m <= forward_m <= p.trigger_max_forward_m
                and abs(right_m) >= p.trigger_min_abs_right_m
            )
            if not severe:
                return actions
            self.active = True
            self.activations += 1

        if visible and forward_m > 0.0:
            self.dropout_ticks = 0
            if (
                abs(right_m) <= p.release_max_abs_right_m
                and abs(yaw_error_rad) <= p.release_max_abs_yaw_error_rad
            ):
                self.releases += 1
                self._deactivate()
                return actions
            current_yaw_rad = _yaw_from_quaternion(values)
            yaw_step_rad = tail._clamp(
                yaw_error_rad,
                -p.max_yaw_step_rad,
                p.max_yaw_step_rad,
            )
            self.last_yaw_target_rad = _wrap_angle(
                current_yaw_rad - yaw_step_rad
            )
        else:
            self.dropout_ticks += 1
            if (
                self.last_yaw_target_rad is None
                or self.dropout_ticks > p.max_dropout_ticks
            ):
                self.dropout_deactivations += 1
                self._deactivate()
                return actions

        self.steps_active += 1
        return [
            p.acquisition_pitch_norm,
            p.acquisition_roll_norm,
            p.acquisition_thrust_norm,
            tail._clamp(float(self.last_yaw_target_rad) / math.pi),
        ]


_CONTROLLER = Gate3SevereAcquisition()


def reset() -> None:
    base.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    actions = base.infer(observation)
    return _CONTROLLER.apply(observation, actions)


def controller_snapshot() -> dict:
    return _CONTROLLER.snapshot()
