#!/usr/bin/env python3
"""N238: persistent observable Gate-4 identity with a bounded intercept.

N237's official traces stayed collision-free but proved that its 0.30-second
hard reset could re-anchor on a different visible gate.  The inherited raw
acquisition action then gave rejected aliases direct roll, thrust, and yaw
authority.  N238 removes both paths.  It owns one camera-relative target
family for the entire official Gate-4 phase, admits only locally coherent or
persistently closer measurements, and never derives an action from a rejected
pose.  Loss of association commands a level brake.  A short forward commit is
available only from a fresh, associated, projected-centered pose.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate4_staged_mpc_n236 as n236


EXPECTED_N236_SHA256 = (
    "fe7984c17b203ade4b3967ff0e6166be17c4d9a89a827231e2e34be6e28862f4"
)
POLICY_STATE_HZ = 60.0
GATE4_INDEX = 3

MAXIMUM_INITIAL_RANGE_M = 40.0
ACQUIRE_CONSECUTIVE_FRESH = 3
REACQUIRE_CONSECUTIVE_FRESH = 3
MAXIMUM_SAMPLE_GAP_S = 0.50
ACQUIRE_MAXIMUM_JUMP_M = 5.0
ASSOCIATION_BASE_JUMP_M = 3.0
ASSOCIATION_SPEED_M_S = 30.0
ASSOCIATION_MAXIMUM_JUMP_M = 12.0
MAXIMUM_FORWARD_REGRESSION_M = 2.5
REACQUIRE_MINIMUM_PROGRESS_M = 3.0
ASSOCIATED_HOLD_S = 0.25

POSITION_FILTER_ALPHA = 0.65
RATE_FILTER_ALPHA = 0.40
MAXIMUM_ABS_RATE_M_S = 20.0
LOOKAHEAD_MAXIMUM_S = 1.50
LOOKAHEAD_CLOSING_FLOOR_M_S = 4.0

BRAKE_PITCH_NORM = 0.25
NEAR_BRAKE_PITCH_NORM = 0.35
FAR_CENTERED_PITCH_NORM = -0.08
COMMIT_PITCH_NORM = -0.18
COMMIT_TRIGGER_FORWARD_M = 8.0
COMMIT_TRIGGER_MAX_ABS_RIGHT_M = 1.0
COMMIT_TRIGGER_MAX_ABS_DOWN_M = 1.0
COMMIT_TRIGGER_MAX_ABS_YAW_RAD = 0.15
COMMIT_DURATION_S = 1.0
NEAR_FORWARD_M = 4.0

ROLL_GAIN_PER_M = 0.07
MAXIMUM_ROLL_NORM = 0.65
THRUST_GAIN_PER_M = 0.06
MAXIMUM_THRUST_NORM = 0.55
MAXIMUM_YAW_STEP_RAD = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_N236_PATH = Path(n236.__file__).resolve()
if _sha256(_N236_PATH) != EXPECTED_N236_SHA256:
    raise RuntimeError("N236 policy source hash mismatch")


def _clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


class Gate4IdentityLockedInterceptController:
    """Associate one observable target family and control only from it."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.gate4_active = False
        self.inference_calls = 0
        self.gate4_calls = 0
        self.last_raw_key: tuple[float, float, float] | None = None
        self.candidate_pose: np.ndarray | None = None
        self.candidate_last_s: float | None = None
        self.candidate_count = 0
        self.associated_pose: np.ndarray | None = None
        self.associated_rate_m_s = np.zeros(3, dtype=np.float64)
        self.associated_yaw_error_rad: float | None = None
        self.associated_last_s: float | None = None
        self.held_yaw_target_rad: float | None = None
        self.accepted_fresh = 0
        self.rejected_aliases = 0
        self.initial_acquisitions = 0
        self.closer_reacquisitions = 0
        self.level_brake_calls = 0
        self.associated_control_calls = 0
        self.commit_triggers = 0
        self.commit_calls = 0
        self.commit_until_s: float | None = None
        self.last_action = (BRAKE_PITCH_NORM, 0.0, 0.0, 0.0)

    @staticmethod
    def _gate(values: np.ndarray) -> int:
        return n236.n235.n234.n233.n232.tail._gate_index(values)

    @staticmethod
    def _pose(values: np.ndarray) -> np.ndarray | None:
        if float(values[10]) < 0.5:
            return None
        pose = np.asarray(
            [
                n236.n235.n234.n233.n232.mpc.inverse_tanh(
                    float(values[11]), 10.0
                ),
                n236.n235.n234.n233.n232.mpc.inverse_tanh(
                    float(values[12]), 5.0
                ),
                n236.n235.n234.n233.n232.mpc.inverse_tanh(
                    float(values[13]), 5.0
                ),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(pose)) or float(pose[0]) <= 0.0:
            return None
        return pose

    @staticmethod
    def _current_yaw_rad(values: np.ndarray) -> float:
        return float(
            n236.n235.n234.n233.n232.mpc.quaternion_to_euler(values[6:10])[2]
        )

    @staticmethod
    def _yaw_error_rad(values: np.ndarray) -> float:
        return _clamp(float(values[14])) * (math.pi / 4.0)

    @staticmethod
    def _candidate_coherent(
        current: np.ndarray,
        previous: np.ndarray | None,
        current_s: float,
        previous_s: float | None,
    ) -> bool:
        if previous is None or previous_s is None:
            return False
        dt = current_s - previous_s
        return (
            0.0 < dt <= MAXIMUM_SAMPLE_GAP_S
            and float(np.linalg.norm(current - previous))
            <= ACQUIRE_MAXIMUM_JUMP_M
        )

    def _advance_candidate(self, pose: np.ndarray, now_s: float) -> None:
        if self._candidate_coherent(
            pose, self.candidate_pose, now_s, self.candidate_last_s
        ):
            self.candidate_count += 1
        else:
            self.candidate_count = 1
        self.candidate_pose = pose.copy()
        self.candidate_last_s = now_s

    def _clear_candidate(self) -> None:
        self.candidate_pose = None
        self.candidate_last_s = None
        self.candidate_count = 0

    def _accept(
        self,
        pose: np.ndarray,
        values: np.ndarray,
        *,
        now_s: float,
        reacquired: bool,
    ) -> None:
        old_pose = self.associated_pose
        old_s = self.associated_last_s
        if old_pose is None or old_s is None or reacquired:
            filtered = pose.copy()
            if reacquired:
                self.associated_rate_m_s.fill(0.0)
        else:
            filtered = old_pose + POSITION_FILTER_ALPHA * (pose - old_pose)
            dt = now_s - old_s
            if dt > 0.0:
                measured = (filtered - old_pose) / dt
                measured = np.clip(
                    measured, -MAXIMUM_ABS_RATE_M_S, MAXIMUM_ABS_RATE_M_S
                )
                self.associated_rate_m_s += RATE_FILTER_ALPHA * (
                    measured - self.associated_rate_m_s
                )
        self.associated_pose = filtered
        self.associated_last_s = now_s
        self.associated_yaw_error_rad = self._yaw_error_rad(values)
        current_yaw = self._current_yaw_rad(values)
        yaw_step = _clamp(
            self.associated_yaw_error_rad,
            -MAXIMUM_YAW_STEP_RAD,
            MAXIMUM_YAW_STEP_RAD,
        )
        self.held_yaw_target_rad = n236.n235.n234.n233.n232.mpc.wrap_angle(
            current_yaw - yaw_step
        )
        self.accepted_fresh += 1
        self._clear_candidate()

    def _observe(self, values: np.ndarray, *, now_s: float) -> bool:
        pose = self._pose(values)
        if pose is None:
            return False
        key = tuple(float(value) for value in pose)
        if key == self.last_raw_key:
            return False
        self.last_raw_key = key

        if self.associated_pose is None:
            if float(np.linalg.norm(pose)) > MAXIMUM_INITIAL_RANGE_M:
                self.rejected_aliases += 1
                self._clear_candidate()
                return False
            self._advance_candidate(pose, now_s)
            if self.candidate_count >= ACQUIRE_CONSECUTIVE_FRESH:
                self._accept(pose, values, now_s=now_s, reacquired=False)
                self.initial_acquisitions += 1
                return True
            return False

        assert self.associated_last_s is not None
        age_s = max(0.0, now_s - self.associated_last_s)
        jump_limit = min(
            ASSOCIATION_MAXIMUM_JUMP_M,
            max(ASSOCIATION_BASE_JUMP_M, ASSOCIATION_SPEED_M_S * age_s),
        )
        jump_m = float(np.linalg.norm(pose - self.associated_pose))
        forward_ok = (
            float(pose[0])
            <= float(self.associated_pose[0]) + MAXIMUM_FORWARD_REGRESSION_M
        )
        closer = (
            float(pose[0])
            <= float(self.associated_pose[0]) - REACQUIRE_MINIMUM_PROGRESS_M
        )
        # Once a discontinuous closer family starts confirmation, a growing
        # time-based jump bound may not promote it early. It must complete the
        # same three-fresh-frame confirmation that began on the first reject.
        if self.candidate_count > 0 and closer:
            self.rejected_aliases += 1
            self._advance_candidate(pose, now_s)
            if self.candidate_count >= REACQUIRE_CONSECUTIVE_FRESH:
                self._accept(pose, values, now_s=now_s, reacquired=True)
                self.closer_reacquisitions += 1
                return True
            return False
        if self.candidate_count > 0:
            self._clear_candidate()
        if jump_m <= jump_limit and forward_ok:
            self._accept(pose, values, now_s=now_s, reacquired=False)
            return True

        self.rejected_aliases += 1
        if not closer:
            self._clear_candidate()
            return False
        self._advance_candidate(pose, now_s)
        if self.candidate_count >= REACQUIRE_CONSECUTIVE_FRESH:
            self._accept(pose, values, now_s=now_s, reacquired=True)
            self.closer_reacquisitions += 1
            return True
        return False

    def _level_brake(self, values: np.ndarray) -> list[float]:
        self.level_brake_calls += 1
        current_yaw = self._current_yaw_rad(values)
        yaw_target = (
            current_yaw
            if self.held_yaw_target_rad is None
            else self.held_yaw_target_rad
        )
        return [BRAKE_PITCH_NORM, 0.0, 0.0, _clamp(yaw_target / math.pi)]

    def _associated_action(self, values: np.ndarray, *, now_s: float) -> list[float]:
        assert self.associated_pose is not None
        assert self.associated_last_s is not None
        age_s = max(0.0, now_s - self.associated_last_s)
        commit_active = (
            self.commit_until_s is not None and now_s <= self.commit_until_s
        )
        if not commit_active and self.commit_until_s is not None:
            self.commit_until_s = None
        if age_s > ASSOCIATED_HOLD_S and not commit_active:
            return self._level_brake(values)

        forward_m, right_m, down_m = (
            float(value) for value in self.associated_pose
        )
        forward_rate_m_s = float(self.associated_rate_m_s[0])
        closing_m_s = max(-forward_rate_m_s, LOOKAHEAD_CLOSING_FLOOR_M_S)
        lookahead_s = min(
            LOOKAHEAD_MAXIMUM_S,
            max(0.0, forward_m) / closing_m_s,
        )
        projected_right_m = right_m + float(self.associated_rate_m_s[1]) * lookahead_s
        projected_down_m = down_m + float(self.associated_rate_m_s[2]) * lookahead_s

        roll = _clamp(
            -ROLL_GAIN_PER_M * projected_right_m,
            -MAXIMUM_ROLL_NORM,
            MAXIMUM_ROLL_NORM,
        )
        thrust = _clamp(
            -THRUST_GAIN_PER_M * projected_down_m,
            -MAXIMUM_THRUST_NORM,
            MAXIMUM_THRUST_NORM,
        )
        yaw_error = 0.0 if self.associated_yaw_error_rad is None else self.associated_yaw_error_rad
        centered = (
            abs(projected_right_m) <= COMMIT_TRIGGER_MAX_ABS_RIGHT_M
            and abs(projected_down_m) <= COMMIT_TRIGGER_MAX_ABS_DOWN_M
            and abs(yaw_error) <= COMMIT_TRIGGER_MAX_ABS_YAW_RAD
        )
        if (
            not commit_active
            and age_s <= ASSOCIATED_HOLD_S
            and 0.0 < forward_m <= COMMIT_TRIGGER_FORWARD_M
            and centered
        ):
            self.commit_until_s = now_s + COMMIT_DURATION_S
            self.commit_triggers += 1
            commit_active = True

        if commit_active:
            pitch = COMMIT_PITCH_NORM
            self.commit_calls += 1
        elif centered:
            pitch = FAR_CENTERED_PITCH_NORM
        elif forward_m <= NEAR_FORWARD_M:
            pitch = NEAR_BRAKE_PITCH_NORM
        else:
            pitch = BRAKE_PITCH_NORM

        yaw_target = (
            self._current_yaw_rad(values)
            if self.held_yaw_target_rad is None
            else self.held_yaw_target_rad
        )
        self.associated_control_calls += 1
        return [pitch, roll, thrust, _clamp(yaw_target / math.pi)]

    def apply(
        self,
        observation: Sequence[float],
        parent_actions: Sequence[float],
        *,
        now_s: float,
    ) -> list[float]:
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (32,):
            raise ValueError(f"expected observation shape (32,), got {values.shape}")
        if len(parent_actions) != 4:
            raise ValueError("expected four parent actions")
        parent = [_clamp(value) for value in parent_actions]
        gate = self._gate(values)
        now = float(now_s)
        self.inference_calls += 1

        if gate != GATE4_INDEX:
            if self.gate4_active:
                self.reset()
                self.inference_calls = 1
            return parent

        self.gate4_active = True
        self.gate4_calls += 1
        self._observe(values, now_s=now)
        if self.associated_pose is None:
            action = self._level_brake(values)
        else:
            action = self._associated_action(values, now_s=now)
        self.last_action = tuple(_clamp(value) for value in action)
        return list(self.last_action)

    def snapshot(self) -> dict:
        return {
            "gate4_phase_active": self.gate4_active,
            "inference_calls": self.inference_calls,
            "gate4_calls": self.gate4_calls,
            "candidate_count": self.candidate_count,
            "associated_pose_body_ned_m": (
                None if self.associated_pose is None else self.associated_pose.tolist()
            ),
            "associated_rate_body_ned_m_s": self.associated_rate_m_s.tolist(),
            "associated_yaw_error_rad": self.associated_yaw_error_rad,
            "associated_last_s": self.associated_last_s,
            "held_yaw_target_rad": self.held_yaw_target_rad,
            "accepted_fresh": self.accepted_fresh,
            "rejected_aliases": self.rejected_aliases,
            "initial_acquisitions": self.initial_acquisitions,
            "closer_reacquisitions": self.closer_reacquisitions,
            "level_brake_calls": self.level_brake_calls,
            "associated_control_calls": self.associated_control_calls,
            "commit_triggers": self.commit_triggers,
            "commit_calls": self.commit_calls,
            "commit_until_s": self.commit_until_s,
            "last_action": self.last_action,
            "contract": {
                "policy_state_hz": POLICY_STATE_HZ,
                "one_target_family_per_gate4_phase": True,
                "maximum_initial_range_m": MAXIMUM_INITIAL_RANGE_M,
                "acquire_consecutive_fresh": ACQUIRE_CONSECUTIVE_FRESH,
                "reacquire_consecutive_fresh": REACQUIRE_CONSECUTIVE_FRESH,
                "maximum_forward_regression_m": MAXIMUM_FORWARD_REGRESSION_M,
                "reacquire_minimum_progress_m": REACQUIRE_MINIMUM_PROGRESS_M,
                "rejected_pose_action_authority": False,
                "association_loss_action": "level_brake",
                "commit_requires_fresh_associated_projected_center": True,
                "runtime_privileged_state": False,
                "gates_1_3_parent_invariant": True,
                "gates_5_6_parent_invariant": True,
            },
        }


_CONTROLLER = Gate4IdentityLockedInterceptController()
_INFERENCE_CALLS = 0


def reset() -> None:
    global _INFERENCE_CALLS
    n236.n235.n234.n233.n232.parent.reset()
    _CONTROLLER.reset()
    _INFERENCE_CALLS = 0


def infer(observation: Sequence[float]) -> list[float]:
    global _INFERENCE_CALLS
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    parent_actions = n236.n235.n234.n233.n232.parent.infer(values)
    logical_time_s = _INFERENCE_CALLS / POLICY_STATE_HZ
    _INFERENCE_CALLS += 1
    return _CONTROLLER.apply(
        values, parent_actions, now_s=logical_time_s
    )


def controller_snapshot() -> dict:
    return {
        "n238_gate4_identity_locked_intercept": _CONTROLLER.snapshot(),
        "parent": n236.n235.n234.n233.n232.parent.controller_snapshot(),
    }
