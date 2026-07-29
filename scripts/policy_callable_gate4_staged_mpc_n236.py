#!/usr/bin/env python3
"""N236: brake-until-centered staging around N235 acquisition MPC."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate4_search_mpc_n235 as n235


EXPECTED_N235_SHA256 = (
    "dd8cd8d89f070c9ca51820c18a9565bb61eb478b1dcd9cce464d4fd4674c569a"
)
POLICY_STATE_HZ = 60.0
CENTER_MAX_ABS_RIGHT_M = 1.5
CENTER_MAX_ABS_DOWN_M = 1.5
CENTER_MAX_ABS_YAW_ERROR_RAD = 0.12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_N235_PATH = Path(n235.__file__).resolve()
if _sha256(_N235_PATH) != EXPECTED_N235_SHA256:
    raise RuntimeError("N235 policy source hash mismatch")


class Gate4StagedMPCController(n235.Gate4SearchMPCController):
    def reset(self) -> None:
        super().reset()
        self.uncentered_brake_calls = 0
        self.centered_forward_calls = 0
        self.blind_brake_override_calls = 0

    def _neutral_action(self, values: np.ndarray) -> list[float]:
        action = super()._neutral_action(values)
        if float(values[10]) < 0.5:
            if action[0] != n235.n234.NEUTRAL_BRAKE_PITCH_NORM:
                self.blind_brake_override_calls += 1
            action[0] = n235.n234.NEUTRAL_BRAKE_PITCH_NORM
            return action

        right_m = n235.n234.n233.n232.mpc.inverse_tanh(float(values[12]), 5.0)
        down_m = n235.n234.n233.n232.mpc.inverse_tanh(float(values[13]), 5.0)
        yaw_error_rad = n235.n234.n233.n232.tail._clamp(float(values[14])) * (
            np.pi / 4.0
        )
        centered = (
            abs(right_m) <= CENTER_MAX_ABS_RIGHT_M
            and abs(down_m) <= CENTER_MAX_ABS_DOWN_M
            and abs(yaw_error_rad) <= CENTER_MAX_ABS_YAW_ERROR_RAD
        )
        if not centered:
            action[0] = n235.n234.NEUTRAL_BRAKE_PITCH_NORM
            self.uncentered_brake_calls += 1
        elif action[0] < 0.0:
            self.centered_forward_calls += 1
        return action

    def snapshot(self) -> dict:
        snapshot = super().snapshot()
        snapshot["staged_acquisition"] = {
            "uncentered_brake_calls": self.uncentered_brake_calls,
            "centered_forward_calls": self.centered_forward_calls,
            "blind_brake_override_calls": self.blind_brake_override_calls,
            "contract": {
                "center_max_abs_right_m": CENTER_MAX_ABS_RIGHT_M,
                "center_max_abs_down_m": CENTER_MAX_ABS_DOWN_M,
                "center_max_abs_yaw_error_rad": CENTER_MAX_ABS_YAW_ERROR_RAD,
                "brake_until_centered": True,
                "blind_search_preserves_yaw_but_brakes_pitch": True,
                "n235_other_actions_unchanged": True,
                "runtime_privileged_state": False,
            },
        }
        return snapshot


_CONTROLLER: Gate4StagedMPCController | None = None
_INFERENCE_CALLS = 0


def _controller() -> Gate4StagedMPCController:
    global _CONTROLLER
    if _CONTROLLER is None:
        verified = n235.n234.n233._load_verified_controller()
        _CONTROLLER = Gate4StagedMPCController(verified.ensemble)
    return _CONTROLLER


def reset() -> None:
    global _INFERENCE_CALLS
    n235.n234.n233.n232.parent.reset()
    _controller().reset()
    _INFERENCE_CALLS = 0


def infer(observation: Sequence[float]) -> list[float]:
    global _INFERENCE_CALLS
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    parent_actions = n235.n234.n233.n232.parent.infer(values)
    logical_time_s = _INFERENCE_CALLS / POLICY_STATE_HZ
    _INFERENCE_CALLS += 1
    return _controller().apply(values, parent_actions, now_s=logical_time_s)


def controller_snapshot() -> dict:
    return {
        "n236_gate4_staged_mpc": _controller().snapshot(),
        "n233_model_validation": {
            "model_validations": n235.n234.n233._MODEL_VALIDATIONS,
            "model_path": (
                None
                if n235.n234.n233._CONTROLLER_MODEL_PATH is None
                else str(n235.n234.n233._CONTROLLER_MODEL_PATH)
            ),
        },
        "parent": n235.n234.n233.n232.parent.controller_snapshot(),
    }
