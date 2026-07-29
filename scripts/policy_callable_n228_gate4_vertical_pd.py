#!/usr/bin/env python3
"""N228: retain N227 lateral control and fix its Gate-4 vertical miss.

N227 reached the associated Gate-4 plane near 2 m right but 4.1 m down. Its
terminal-rate forecast reduced descent too early, then the inherited controller
fell back to minimum thrust after the predicted plane. This wrapper uses direct
observable down-position/rate damping for thrust and bounds the post-plane miss
response. No native position, collision state, or gate coordinates are used.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_n227_gate4_terminal_pd as n227
import policy_callable_six_gate_composite as tail


EXPECTED_N227_SOURCE_SHA256 = (
    "f0fd8ed0017d8130da7bbab40e07f6c9eacdf6f8aa9362582661ad6a6ec2bde4"
)
_N227_PATH = Path(n227.__file__).resolve()
if hashlib.sha256(_N227_PATH.read_bytes()).hexdigest() != EXPECTED_N227_SOURCE_SHA256:
    raise RuntimeError("N227 policy source hash mismatch")


GATE4_INDEX = 3
THRUST_DOWN_KP = 0.15
THRUST_DOWN_RATE_KD = 0.10
POST_PLANE_RECOVERY_LIMIT_M = -5.0
POST_PLANE_ROLL_RIGHT_KP = 0.18
POST_PLANE_ROLL_RIGHT_RATE_KD = 0.08


class Gate4VerticalPdController:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active_samples = 0
        self.changed_samples = 0
        self.post_plane_samples = 0
        self.miss_limited_samples = 0
        self.last_down_m: float | None = None
        self.last_down_rate_m_s: float | None = None

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
            self.last_down_m = None
            self.last_down_rate_m_s = None
            return actions

        forward_m = tail._inverse_tanh_norm(float(values[11]), 10.0)
        if forward_m <= POST_PLANE_RECOVERY_LIMIT_M:
            if actions[1] != 0.0 or actions[2] != 0.0:
                self.changed_samples += 1
            actions[1] = 0.0
            actions[2] = 0.0
            self.active_samples += 1
            self.miss_limited_samples += 1
            return actions

        down_m = tail._inverse_tanh_norm(float(values[13]), 5.0)
        down_rate_m_s = tail._inverse_tanh_norm(float(values[2]), 3.0)
        governed_thrust = tail._clamp(
            -THRUST_DOWN_KP * down_m - THRUST_DOWN_RATE_KD * down_rate_m_s
        )
        governed_roll = actions[1]
        if forward_m <= 0.0:
            right_m = tail._inverse_tanh_norm(float(values[12]), 5.0)
            right_rate_m_s = tail._inverse_tanh_norm(float(values[1]), 3.0)
            governed_roll = tail._clamp(
                -POST_PLANE_ROLL_RIGHT_KP * right_m
                - POST_PLANE_ROLL_RIGHT_RATE_KD * right_rate_m_s
            )
            self.post_plane_samples += 1
        if governed_roll != actions[1] or governed_thrust != actions[2]:
            self.changed_samples += 1
        actions[1] = governed_roll
        actions[2] = governed_thrust
        self.active_samples += 1
        self.last_down_m = down_m
        self.last_down_rate_m_s = down_rate_m_s
        return actions

    def snapshot(self) -> dict:
        return {
            "active_samples": self.active_samples,
            "changed_samples": self.changed_samples,
            "post_plane_samples": self.post_plane_samples,
            "miss_limited_samples": self.miss_limited_samples,
            "last_down_m": self.last_down_m,
            "last_down_rate_m_s": self.last_down_rate_m_s,
            "parameters": {
                "gate_index": GATE4_INDEX,
                "thrust_down_kp": THRUST_DOWN_KP,
                "thrust_down_rate_kd": THRUST_DOWN_RATE_KD,
                "post_plane_recovery_limit_m": POST_PLANE_RECOVERY_LIMIT_M,
                "preserve_n227_roll_before_plane": True,
                "runtime_privileged_state": False,
            },
        }


_CONTROLLER = Gate4VerticalPdController()


def reset() -> None:
    n227.reset()
    _CONTROLLER.reset()


def infer(observation: Sequence[float]) -> list[float]:
    return _CONTROLLER.apply(observation, n227.infer(observation))


def controller_snapshot() -> dict:
    return {
        "n228_gate4_vertical_pd": _CONTROLLER.snapshot(),
        "n227": n227.controller_snapshot(),
    }
