#!/usr/bin/env python3
"""N234: fixed-clock Gate-4 MPC with bounded track reacquisition.

N233's official traces proved that its optimizer was accepted only twice per
Gate-4 activation. Most decisions came from the optimizer's aggressive stale
or uncertain-state failsafe, and the last saturated action could persist after
the target disappeared. N234 preserves the exact N233 parent and model, but
requires three coherent fresh poses, resets a stale association, and emits a
neutral brake/hover action whenever no state or no accepted plan is available.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate4_mpc_n233 as n233


EXPECTED_N233_SHA256 = (
    "4a3f0c7a0c8b73053f6aa7da8995b4fffe1e2c8ed2f430c79135476ca8f1c4ba"
)
POLICY_STATE_HZ = 60.0
MAXIMUM_INITIAL_RANGE_M = 75.0
REQUIRED_COHERENT_FRESH_SAMPLES = 3
TRACK_STALE_RESET_S = 0.30
NEUTRAL_BRAKE_PITCH_NORM = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_N233_PATH = Path(n233.__file__).resolve()
if _sha256(_N233_PATH) != EXPECTED_N233_SHA256:
    raise RuntimeError("N233 policy source hash mismatch")


class Gate4ReacquiringMPCController:
    def __init__(self, ensemble: n233.n232.mpc.VisualDynamicsEnsemble) -> None:
        self.ensemble = ensemble
        tracker_config = dataclasses.replace(
            n233.n232.mpc.TrackerConfig(),
            maximum_initial_range_m=MAXIMUM_INITIAL_RANGE_M,
        )
        self.tracker = n233.n232.mpc.Gate4VisualStateTracker(tracker_config)
        self.optimizer = n233.n232.mpc.Gate4RecedingHorizonOptimizer(ensemble)
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self.optimizer.reset()
        self.gate4_phase_active = False
        self.yaw_target_rad: float | None = None
        self.cached_action: tuple[float, float, float, float] | None = None
        self.last_plan_s: float | None = None
        self.last_plan: n233.n232.mpc.PlanResult | None = None
        self.coherent_fresh_samples = 0
        self.inference_calls = 0
        self.gate4_calls = 0
        self.optimized_plans = 0
        self.rejected_plans = 0
        self.cached_action_calls = 0
        self.acquisition_wait_calls = 0
        self.neutral_action_calls = 0
        self.track_resets = 0
        self.plan_rejection_counts: collections.Counter[str] = collections.Counter()

    @staticmethod
    def _base_actions(values: Sequence[float]) -> list[float]:
        if len(values) != 4:
            raise ValueError("expected four base actions")
        return [n233.n232.tail._clamp(float(value)) for value in values]

    @staticmethod
    def _current_yaw_action(values: np.ndarray) -> float:
        quaternion = np.asarray(values[6:10], dtype=np.float64)
        yaw = float(n233.n232.mpc.quaternion_to_euler(quaternion)[2])
        return n233.n232.tail._clamp(yaw / math.pi)

    def _neutral_action(self, values: np.ndarray) -> list[float]:
        self.neutral_action_calls += 1
        return [
            NEUTRAL_BRAKE_PITCH_NORM,
            0.0,
            0.0,
            self._current_yaw_action(values),
        ]

    def _leave_gate4(self) -> None:
        if self.gate4_phase_active:
            self.tracker.leave_gate4()
        self.gate4_phase_active = False
        self.yaw_target_rad = None
        self.cached_action = None
        self.last_plan_s = None
        self.last_plan = None
        self.coherent_fresh_samples = 0
        self.optimizer.reset()

    def _reset_stale_track(self, values: np.ndarray) -> None:
        velocity = self.tracker._vehicle_velocity_from_gate_rate(values)
        self.tracker.reset()
        self.tracker.carried_velocity_world_m_s = velocity
        self.optimizer.reset()
        self.yaw_target_rad = None
        self.cached_action = None
        self.last_plan_s = None
        self.last_plan = None
        self.coherent_fresh_samples = 0
        self.track_resets += 1

    def apply(
        self,
        observation: Sequence[float],
        base_actions: Sequence[float],
        *,
        now_s: float,
    ) -> list[float]:
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (32,):
            raise ValueError(f"expected observation shape (32,), got {values.shape}")
        actions = self._base_actions(base_actions)
        gate = n233.n232.tail._gate_index(values)
        now = float(now_s)
        self.inference_calls += 1

        if gate == n233.n232.GATE3_INDEX:
            self.tracker.observe_prefix(values, gate_index=gate)
            self._leave_gate4()
            return actions
        if gate != n233.n232.GATE4_INDEX:
            self._leave_gate4()
            return actions

        self.gate4_calls += 1
        self.gate4_phase_active = True
        rejected_before = self.tracker.rejected_samples
        state, fresh = self.tracker.update_gate4(values, now_s=now)
        if state is not None and state.measurement_age_s > TRACK_STALE_RESET_S:
            self._reset_stale_track(values)
            rejected_before = self.tracker.rejected_samples
            state, fresh = self.tracker.update_gate4(values, now_s=now)

        if fresh:
            self.coherent_fresh_samples += 1
        elif self.tracker.rejected_samples > rejected_before:
            self.coherent_fresh_samples = 0

        if (
            state is None
            or self.coherent_fresh_samples < REQUIRED_COHERENT_FRESH_SAMPLES
        ):
            self.acquisition_wait_calls += 1
            self.cached_action = tuple(self._neutral_action(values))
            return list(self.cached_action)

        if self.yaw_target_rad is None:
            current_yaw = float(state.euler_rpy_rad[2])
            yaw_error = n233.n232.tail._clamp(float(values[14])) * (math.pi / 4.0)
            yaw_step = n233.n232.mpc.clamp(
                yaw_error,
                -n233.n232.MAXIMUM_YAW_STEP_RAD,
                n233.n232.MAXIMUM_YAW_STEP_RAD,
            )
            self.yaw_target_rad = n233.n232.mpc.wrap_angle(current_yaw - yaw_step)

        time_since_plan_s = (
            math.inf if self.last_plan_s is None else now - self.last_plan_s
        )
        replan = (
            self.cached_action is None
            or self.last_plan_s is None
            or time_since_plan_s >= n233.n232.REPLAN_MAX_INTERVAL_S
            or (
                fresh
                and time_since_plan_s >= n233.n232.REPLAN_MIN_INTERVAL_S
            )
        )
        if replan:
            assert self.yaw_target_rad is not None
            plan = self.optimizer.plan(state, yaw_target_rad=self.yaw_target_rad)
            self.last_plan = plan
            self.last_plan_s = now
            if plan.accepted:
                self.cached_action = plan.action
                self.optimized_plans += 1
            else:
                self.cached_action = tuple(self._neutral_action(values))
                self.rejected_plans += 1
                self.plan_rejection_counts[plan.reason] += 1
        else:
            self.cached_action_calls += 1
        assert self.cached_action is not None
        return [n233.n232.tail._clamp(value) for value in self.cached_action]

    def snapshot(self) -> dict:
        return {
            "gate4_phase_active": self.gate4_phase_active,
            "yaw_target_rad": self.yaw_target_rad,
            "cached_action": self.cached_action,
            "last_plan_s": self.last_plan_s,
            "last_plan": (
                None if self.last_plan is None else dataclasses.asdict(self.last_plan)
            ),
            "coherent_fresh_samples": self.coherent_fresh_samples,
            "inference_calls": self.inference_calls,
            "gate4_calls": self.gate4_calls,
            "optimized_plans": self.optimized_plans,
            "rejected_plans": self.rejected_plans,
            "cached_action_calls": self.cached_action_calls,
            "acquisition_wait_calls": self.acquisition_wait_calls,
            "neutral_action_calls": self.neutral_action_calls,
            "track_resets": self.track_resets,
            "plan_rejection_counts": dict(sorted(self.plan_rejection_counts.items())),
            "tracker": self.tracker.snapshot(),
            "optimizer_config": dataclasses.asdict(self.optimizer.config),
            "contract": {
                "policy_state_hz": POLICY_STATE_HZ,
                "maximum_initial_range_m": MAXIMUM_INITIAL_RANGE_M,
                "required_coherent_fresh_samples": REQUIRED_COHERENT_FRESH_SAMPLES,
                "track_stale_reset_s": TRACK_STALE_RESET_S,
                "neutral_on_missing_or_rejected_plan": True,
                "gates_1_3_parent_invariant": True,
                "gates_5_6_parent_invariant": True,
                "n233_parent_and_model_unchanged": True,
                "runtime_privileged_state": False,
            },
        }


_CONTROLLER: Gate4ReacquiringMPCController | None = None
_INFERENCE_CALLS = 0


def _controller() -> Gate4ReacquiringMPCController:
    global _CONTROLLER
    if _CONTROLLER is None:
        verified = n233._load_verified_controller()
        _CONTROLLER = Gate4ReacquiringMPCController(verified.ensemble)
    return _CONTROLLER


def reset() -> None:
    global _INFERENCE_CALLS
    n233.n232.parent.reset()
    _controller().reset()
    _INFERENCE_CALLS = 0


def infer(observation: Sequence[float]) -> list[float]:
    global _INFERENCE_CALLS
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    parent_actions = n233.n232.parent.infer(values)
    logical_time_s = _INFERENCE_CALLS / POLICY_STATE_HZ
    _INFERENCE_CALLS += 1
    return _controller().apply(values, parent_actions, now_s=logical_time_s)


def controller_snapshot() -> dict:
    return {
        "n234_gate4_reacquiring_mpc": _controller().snapshot(),
        "n233_model_validation": {
            "model_validations": n233._MODEL_VALIDATIONS,
            "model_path": (
                None
                if n233._CONTROLLER_MODEL_PATH is None
                else str(n233._CONTROLLER_MODEL_PATH)
            ),
        },
        "parent": n233.n232.parent.controller_snapshot(),
    }
