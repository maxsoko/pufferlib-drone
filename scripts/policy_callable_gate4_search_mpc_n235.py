#!/usr/bin/env python3
"""N235: N234 with bounded observable Gate-4 acquisition and search.

N234 removed stale saturated actions, but its one live Gate-4 activation held
brake/hover while a raw target moved out of frame. N235 keeps the same admitted
MPC path and replaces only the no-plan action with low-gain raw visual
acquisition. A missing target is searched in its last yaw direction for at
most two seconds, after which the controller returns to brake/hover.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate4_reacquiring_mpc_n234 as n234


EXPECTED_N234_SHA256 = (
    "84c08dac68be66039567173a7319831232601c22790792588a044401d7671530"
)
POLICY_STATE_HZ = 60.0
ACQUISITION_FAR_FORWARD_M = 12.0
ACQUISITION_NEAR_FORWARD_M = 6.0
ACQUISITION_FORWARD_PITCH_NORM = -0.12
ACQUISITION_ROLL_GAIN_PER_M = 0.03
ACQUISITION_MAX_ROLL_NORM = 0.55
ACQUISITION_THRUST_GAIN_PER_M = 0.04
ACQUISITION_MAX_THRUST_NORM = 0.50
ACQUISITION_MAX_YAW_STEP_RAD = 0.25
BLIND_SEARCH_YAW_STEP_RAD = 0.10
BLIND_SEARCH_DURATION_S = 2.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_N234_PATH = Path(n234.__file__).resolve()
if _sha256(_N234_PATH) != EXPECTED_N234_SHA256:
    raise RuntimeError("N234 policy source hash mismatch")


class Gate4SearchMPCController(n234.Gate4ReacquiringMPCController):
    def reset(self) -> None:
        super().reset()
        self.control_now_s = 0.0
        self.last_visible_s: float | None = None
        self.last_yaw_direction = 0.0
        self.visible_acquisition_calls = 0
        self.blind_search_calls = 0
        self.blind_brake_calls = 0

    def _leave_gate4(self) -> None:
        super()._leave_gate4()
        self.last_visible_s = None
        self.last_yaw_direction = 0.0

    def _neutral_action(self, values: np.ndarray) -> list[float]:
        self.neutral_action_calls += 1
        current_yaw = float(
            n234.n233.n232.mpc.quaternion_to_euler(values[6:10])[2]
        )
        visible = float(values[10]) >= 0.5
        if visible:
            forward_m = n234.n233.n232.mpc.inverse_tanh(float(values[11]), 10.0)
            right_m = n234.n233.n232.mpc.inverse_tanh(float(values[12]), 5.0)
            down_m = n234.n233.n232.mpc.inverse_tanh(float(values[13]), 5.0)
            yaw_error = n234.n233.n232.tail._clamp(float(values[14])) * (
                math.pi / 4.0
            )
            if abs(yaw_error) > 1e-4:
                self.last_yaw_direction = math.copysign(1.0, yaw_error)
            self.last_visible_s = self.control_now_s
            yaw_step = n234.n233.n232.mpc.clamp(
                yaw_error,
                -ACQUISITION_MAX_YAW_STEP_RAD,
                ACQUISITION_MAX_YAW_STEP_RAD,
            )
            desired_yaw = n234.n233.n232.mpc.wrap_angle(current_yaw - yaw_step)
            if forward_m > ACQUISITION_FAR_FORWARD_M:
                pitch = ACQUISITION_FORWARD_PITCH_NORM
            elif forward_m > ACQUISITION_NEAR_FORWARD_M:
                pitch = 0.0
            else:
                pitch = n234.NEUTRAL_BRAKE_PITCH_NORM
            roll = n234.n233.n232.mpc.clamp(
                -ACQUISITION_ROLL_GAIN_PER_M * right_m,
                -ACQUISITION_MAX_ROLL_NORM,
                ACQUISITION_MAX_ROLL_NORM,
            )
            thrust = n234.n233.n232.mpc.clamp(
                -ACQUISITION_THRUST_GAIN_PER_M * down_m,
                -ACQUISITION_MAX_THRUST_NORM,
                ACQUISITION_MAX_THRUST_NORM,
            )
            self.visible_acquisition_calls += 1
            return [pitch, roll, thrust, desired_yaw / math.pi]

        blind_age_s = (
            math.inf
            if self.last_visible_s is None
            else self.control_now_s - self.last_visible_s
        )
        if blind_age_s <= BLIND_SEARCH_DURATION_S and self.last_yaw_direction != 0.0:
            desired_yaw = n234.n233.n232.mpc.wrap_angle(
                current_yaw - self.last_yaw_direction * BLIND_SEARCH_YAW_STEP_RAD
            )
            self.blind_search_calls += 1
            return [ACQUISITION_FORWARD_PITCH_NORM, 0.0, 0.0, desired_yaw / math.pi]

        self.blind_brake_calls += 1
        return [
            n234.NEUTRAL_BRAKE_PITCH_NORM,
            0.0,
            0.0,
            current_yaw / math.pi,
        ]

    def apply(
        self,
        observation: Sequence[float],
        base_actions: Sequence[float],
        *,
        now_s: float,
    ) -> list[float]:
        self.control_now_s = float(now_s)
        return super().apply(observation, base_actions, now_s=now_s)

    def snapshot(self) -> dict:
        snapshot = super().snapshot()
        snapshot["acquisition_search"] = {
            "control_now_s": self.control_now_s,
            "last_visible_s": self.last_visible_s,
            "last_yaw_direction": self.last_yaw_direction,
            "visible_acquisition_calls": self.visible_acquisition_calls,
            "blind_search_calls": self.blind_search_calls,
            "blind_brake_calls": self.blind_brake_calls,
            "contract": {
                "far_forward_m": ACQUISITION_FAR_FORWARD_M,
                "near_forward_m": ACQUISITION_NEAR_FORWARD_M,
                "forward_pitch_norm": ACQUISITION_FORWARD_PITCH_NORM,
                "roll_gain_per_m": ACQUISITION_ROLL_GAIN_PER_M,
                "maximum_roll_norm": ACQUISITION_MAX_ROLL_NORM,
                "thrust_gain_per_m": ACQUISITION_THRUST_GAIN_PER_M,
                "maximum_thrust_norm": ACQUISITION_MAX_THRUST_NORM,
                "maximum_yaw_step_rad": ACQUISITION_MAX_YAW_STEP_RAD,
                "blind_search_yaw_step_rad": BLIND_SEARCH_YAW_STEP_RAD,
                "blind_search_duration_s": BLIND_SEARCH_DURATION_S,
                "runtime_privileged_state": False,
            },
        }
        snapshot["contract"]["n234_admitted_mpc_unchanged"] = True
        return snapshot


_CONTROLLER: Gate4SearchMPCController | None = None
_INFERENCE_CALLS = 0


def _controller() -> Gate4SearchMPCController:
    global _CONTROLLER
    if _CONTROLLER is None:
        verified = n234.n233._load_verified_controller()
        _CONTROLLER = Gate4SearchMPCController(verified.ensemble)
    return _CONTROLLER


def reset() -> None:
    global _INFERENCE_CALLS
    n234.n233.n232.parent.reset()
    _controller().reset()
    _INFERENCE_CALLS = 0


def infer(observation: Sequence[float]) -> list[float]:
    global _INFERENCE_CALLS
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    parent_actions = n234.n233.n232.parent.infer(values)
    logical_time_s = _INFERENCE_CALLS / POLICY_STATE_HZ
    _INFERENCE_CALLS += 1
    return _controller().apply(values, parent_actions, now_s=logical_time_s)


def controller_snapshot() -> dict:
    return {
        "n235_gate4_search_mpc": _controller().snapshot(),
        "n233_model_validation": {
            "model_validations": n234.n233._MODEL_VALIDATIONS,
            "model_path": (
                None
                if n234.n233._CONTROLLER_MODEL_PATH is None
                else str(n234.n233._CONTROLLER_MODEL_PATH)
            ),
        },
        "parent": n234.n233.n232.parent.controller_snapshot(),
    }
