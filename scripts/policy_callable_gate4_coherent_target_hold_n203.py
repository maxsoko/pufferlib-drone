#!/usr/bin/env python3
"""N205: exact H12/N203 plus coherent rejected-visible Gate-4 yaw hold.

The Gate-4 association filter can reject a visible pose after it has acquired a
coherent anchor.  The parent nevertheless steers yaw from every visible pose,
including rejected ones.  This wrapper keeps the most recent yaw target that
was produced by an associated pose while a visible pose is rejected.  It does
not change acquisition before the first anchor, accepted/re-anchored samples,
dropout behavior, or pitch/roll/thrust.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate3_counter_hysteresis_n202 as n203
import policy_callable_six_gate_composite as tail
import policy_callable_six_gate_hybrid as hybrid


EXPECTED_N203_SHA256 = (
    "b357536767a62beb85ba707a297b67ef5191998a425f6e0912a9882e07ee5cbe"
)
_N203_PATH = Path(n203.__file__).resolve()
if hashlib.sha256(_N203_PATH.read_bytes()).hexdigest() != EXPECTED_N203_SHA256:
    raise RuntimeError("N203 policy source hash mismatch")


ASSOCIATION_EPSILON_M = 1e-4
MAX_HELD_YAW_DELTA_RAD = hybrid.GATE4_ACQUISITION_MAX_YAW_STEP_RAD


def _associated_with_current_observation(values: np.ndarray) -> bool:
    current_vector = hybrid._gate4_vector(values)
    accepted_vector = hybrid._GATE4_LAST_VECTOR
    return (
        current_vector is not None
        and accepted_vector is not None
        and hybrid._distance_m(current_vector, accepted_vector)
        <= ASSOCIATION_EPSILON_M
    )


class Gate4CoherentTargetHold:
    """Reject yaw updates from poses rejected by the target association."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.held_samples = 0
        self.last_held_target_rad: float | None = None
        self.last_commanded_yaw_rad: float | None = None

    def snapshot(self) -> dict:
        return {
            "held_samples": self.held_samples,
            "last_held_target_rad": self.last_held_target_rad,
            "last_commanded_yaw_rad": self.last_commanded_yaw_rad,
            "parameters": {
                "gate_index": 3,
                "association_epsilon_m": ASSOCIATION_EPSILON_M,
                "max_held_yaw_delta_rad": MAX_HELD_YAW_DELTA_RAD,
                "requires_previous_anchor": True,
                "visible_rejected_only": True,
                "preserve_pitch_roll_thrust": True,
                "pre_anchor_acquisition_passthrough": True,
                "associated_pose_passthrough": True,
                "dropout_passthrough": True,
            },
        }

    def apply(
        self,
        observation: Sequence[float],
        base_actions: Sequence[float],
        *,
        had_anchor: bool,
        previous_coherent_yaw_target_rad: float | None,
        current_pose_associated: bool,
    ) -> list[float]:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (32,):
            raise ValueError(f"expected observation shape (32,), got {values.shape}")
        if len(base_actions) != 4:
            raise ValueError("expected four base actions")
        actions = [tail._clamp(float(value)) for value in base_actions]
        should_hold = (
            tail._gate_index(values) == 3
            and float(values[10]) > 0.5
            and had_anchor
            and previous_coherent_yaw_target_rad is not None
            and not current_pose_associated
        )
        if not should_hold:
            return actions

        current_yaw_rad = hybrid._yaw_from_quaternion(values)
        yaw_delta_rad = tail._clamp(
            hybrid._wrap_angle(
                float(previous_coherent_yaw_target_rad) - current_yaw_rad
            ),
            -MAX_HELD_YAW_DELTA_RAD,
            MAX_HELD_YAW_DELTA_RAD,
        )
        commanded_yaw_rad = hybrid._wrap_angle(current_yaw_rad + yaw_delta_rad)
        actions[3] = tail._clamp(commanded_yaw_rad / math.pi)
        self.held_samples += 1
        self.last_held_target_rad = float(previous_coherent_yaw_target_rad)
        self.last_commanded_yaw_rad = commanded_yaw_rad
        return actions


_CONTROLLER = Gate4CoherentTargetHold()


def reset() -> None:
    n203.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    values = np.asarray(observation, dtype=np.float32)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")

    had_anchor = hybrid._GATE4_ANCHOR_VECTOR is not None
    previous_target = hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_RAD
    actions = n203.infer(values)
    associated = _associated_with_current_observation(values)
    held_samples_before = _CONTROLLER.held_samples
    held_actions = _CONTROLLER.apply(
        values,
        actions,
        had_anchor=had_anchor,
        previous_coherent_yaw_target_rad=previous_target,
        current_pose_associated=associated,
    )
    if _CONTROLLER.held_samples > held_samples_before:
        # Keep the parent's dropout carry coherent with the output actually
        # commanded here.  Refresh only while a rejected pose remains visible.
        hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_RAD = previous_target
        hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_S = time.monotonic()
    return held_actions


def controller_snapshot() -> dict:
    return {
        "gate4_coherent_target_hold": _CONTROLLER.snapshot(),
        "n203": n203.controller_snapshot(),
    }
