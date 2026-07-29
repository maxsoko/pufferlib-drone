#!/usr/bin/env python3
"""N207: N206 learned Gate-4 tail plus one bounded terminal crossing pulse.

Stage-5 run 1 reached Gate 4 centered inside 1.5 m, but the learned checkpoint
commanded negative pitch and contacted the gate without an official crossing.
This wrapper changes only that terminal case.  It uses the current observable
gate pose to trigger one short forward-pitch floor, preserves learned roll,
thrust, and yaw, and cannot retrigger during the same Gate-4 phase.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate4_learned_tail_n203 as n206
import policy_callable_six_gate_composite as tail


EXPECTED_N206_SHA256 = (
    "4a976df88dc95af46d2bd022fd8497608ff9cbe3a47c2265c4215e8243e775af"
)
_N206_PATH = Path(n206.__file__).resolve()
if hashlib.sha256(_N206_PATH.read_bytes()).hexdigest() != EXPECTED_N206_SHA256:
    raise RuntimeError("N206 policy source hash mismatch")


GATE4_INDEX = 3
MAX_FORWARD_M = 3.0
MAX_ABS_RIGHT_M = 0.5
MAX_ABS_DOWN_M = 0.5
MAX_ABS_YAW_ERROR_RAD = 0.15
FORWARD_PITCH_RAD = 0.06
FORWARD_PITCH_NORM = FORWARD_PITCH_RAD / 0.5
PULSE_DURATION_S = 0.5


class Gate4TerminalCrossingPulse:
    """Apply one short pitch floor after a centered observable Gate-4 pose."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.triggered = False
        self.active_until_s: float | None = None
        self.trigger_count = 0
        self.active_samples = 0
        self.changed_samples = 0
        self.last_trigger_pose: dict[str, float] | None = None

    def _qualified_pose(self, values: np.ndarray) -> dict[str, float] | None:
        if float(values[10]) <= 0.5:
            return None
        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
        down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
        yaw_error_rad = tail._clamp(float(values[14])) * (math.pi / 4.0)
        if not (
            0.0 < forward_m <= MAX_FORWARD_M
            and abs(right_m) <= MAX_ABS_RIGHT_M
            and abs(down_m) <= MAX_ABS_DOWN_M
            and abs(yaw_error_rad) <= MAX_ABS_YAW_ERROR_RAD
        ):
            return None
        return {
            "forward_m": forward_m,
            "right_m": right_m,
            "down_m": down_m,
            "yaw_error_rad": yaw_error_rad,
        }

    def apply(
        self,
        observation: Sequence[float],
        base_actions: Sequence[float],
        *,
        now_s: float | None = None,
    ) -> list[float]:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (32,):
            raise ValueError(f"expected observation shape (32,), got {values.shape}")
        if len(base_actions) != 4:
            raise ValueError("expected four base actions")
        actions = [tail._clamp(float(value)) for value in base_actions]

        if tail._gate_index(values) != GATE4_INDEX:
            self.reset()
            return actions

        now = time.monotonic() if now_s is None else float(now_s)
        pulse_active = (
            self.active_until_s is not None and now <= self.active_until_s
        )
        if not pulse_active and self.active_until_s is not None:
            self.active_until_s = None

        if not self.triggered:
            pose = self._qualified_pose(values)
            if pose is not None:
                self.triggered = True
                self.trigger_count += 1
                self.last_trigger_pose = pose
                self.active_until_s = now + PULSE_DURATION_S
                pulse_active = True

        if pulse_active:
            pitch_before = actions[0]
            actions[0] = max(pitch_before, FORWARD_PITCH_NORM)
            self.active_samples += 1
            if actions[0] != pitch_before:
                self.changed_samples += 1
        return actions

    def snapshot(self) -> dict:
        return {
            "triggered": self.triggered,
            "active_until_s": self.active_until_s,
            "trigger_count": self.trigger_count,
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "last_trigger_pose": self.last_trigger_pose,
            "parameters": {
                "gate_index": GATE4_INDEX,
                "max_forward_m": MAX_FORWARD_M,
                "max_abs_right_m": MAX_ABS_RIGHT_M,
                "max_abs_down_m": MAX_ABS_DOWN_M,
                "max_abs_yaw_error_rad": MAX_ABS_YAW_ERROR_RAD,
                "forward_pitch_rad": FORWARD_PITCH_RAD,
                "forward_pitch_norm": FORWARD_PITCH_NORM,
                "pulse_duration_s": PULSE_DURATION_S,
                "one_shot_per_gate4_phase": True,
                "dropout_carry_only_while_pulse_active": True,
                "preserve_roll_thrust_yaw": True,
                "course_coordinates_used": False,
                "parent": "N206_learned_gate4_tail",
            },
        }


_CONTROLLER = Gate4TerminalCrossingPulse()


def reset() -> None:
    n206.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    return _CONTROLLER.apply(values, n206.infer(values))


def controller_snapshot() -> dict:
    return {
        "gate4_terminal_crossing_pulse": _CONTROLLER.snapshot(),
        "n206": n206.controller_snapshot(),
    }
