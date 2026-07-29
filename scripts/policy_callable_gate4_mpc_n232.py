#!/usr/bin/env python3
"""N232: live-rate parent plus bounded visual Gate-4 MPC.

Gates 1--3 and 5--6 are byte-for-byte parent actions. During Gate 3 the
wrapper passively carries the observable gate-relative velocity into Gate 4.
At Gate 4 it accepts only current coherent raw poses, replans a short
pitch/roll/thrust sequence against the official-trace dynamics ensemble, and
executes only the first action.  Prediction stops at the gate plane, two
seconds, 0.30 seconds of measurement age, or the uncertainty bound. The
wrapper's deterministic 10 Hz logical clock advances once per completed live
inference; the runner must not attempt impossible wall-clock catch-up steps.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import gate4_receding_horizon as mpc
import policy_callable_gate4_terminal_crossing_n206 as parent
import policy_callable_six_gate_composite as tail


EXPECTED_PARENT_SHA256 = (
    "b4ccbcb90f30ce866c8a9f827c98d457708ca925345cd1847a6d3ef3ebf80f63"
)
EXPECTED_MPC_SOURCE_SHA256 = (
    "3f0fe8f12cca5b5b9b9bc647ee15702379531c059360baed504b24882a7a7550"
)
EXPECTED_MODEL_SHA256 = (
    "9aaefaae111734314353757adbd49da913f13c9bb558627e382ba3c2eb8cc275"
)


def _assert_source_hash(module, expected: str, label: str) -> None:
    path = Path(module.__file__).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{label} source hash mismatch: expected {expected}, got {actual}"
        )


_assert_source_hash(parent, EXPECTED_PARENT_SHA256, "N231 parent")
_assert_source_hash(mpc, EXPECTED_MPC_SOURCE_SHA256, "N232 MPC")


GATE3_INDEX = 2
GATE4_INDEX = 3
MAXIMUM_YAW_STEP_RAD = 0.12
REPLAN_MIN_INTERVAL_S = 0.18
REPLAN_MAX_INTERVAL_S = 0.25
POLICY_LOGICAL_HZ = 10.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_model_path() -> Path:
    value = os.getenv("PUFFER_POLICY_GATE4_DYNAMICS_PATH", "").strip()
    if not value:
        raise RuntimeError("PUFFER_POLICY_GATE4_DYNAMICS_PATH is required")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Gate-4 dynamics hash mismatch: "
            f"expected {EXPECTED_MODEL_SHA256}, got {actual}"
        )
    return path


class Gate4MPCController:
    def __init__(self, ensemble: mpc.VisualDynamicsEnsemble) -> None:
        self.ensemble = ensemble
        self.tracker = mpc.Gate4VisualStateTracker()
        self.optimizer = mpc.Gate4RecedingHorizonOptimizer(ensemble)
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self.optimizer.reset()
        self.gate4_phase_active = False
        self.yaw_target_rad: float | None = None
        self.cached_action: tuple[float, float, float, float] | None = None
        self.last_plan_s: float | None = None
        self.last_plan: mpc.PlanResult | None = None
        self.inference_calls = 0
        self.gate4_calls = 0
        self.optimized_plans = 0
        self.failsafe_plans = 0
        self.cached_action_calls = 0

    @staticmethod
    def _base_actions(values: Sequence[float]) -> list[float]:
        if len(values) != 4:
            raise ValueError("expected four base actions")
        return [tail._clamp(float(value)) for value in values]

    def _leave_gate4(self) -> None:
        if self.gate4_phase_active:
            self.tracker.leave_gate4()
        self.gate4_phase_active = False
        self.yaw_target_rad = None
        self.cached_action = None
        self.last_plan_s = None
        self.last_plan = None
        self.optimizer.reset()

    def apply(
        self,
        observation: Sequence[float],
        base_actions: Sequence[float],
        *,
        now_s: float | None = None,
    ) -> list[float]:
        values = np.asarray(observation, dtype=np.float64)
        if values.shape != (32,):
            raise ValueError(f"expected observation shape (32,), got {values.shape}")
        actions = self._base_actions(base_actions)
        gate = tail._gate_index(values)
        now = (
            self.inference_calls / POLICY_LOGICAL_HZ
            if now_s is None
            else float(now_s)
        )
        self.inference_calls += 1

        if gate == GATE3_INDEX:
            self.tracker.observe_prefix(values, gate_index=gate)
            self._leave_gate4()
            return actions
        if gate != GATE4_INDEX:
            self._leave_gate4()
            return actions

        self.gate4_calls += 1
        self.gate4_phase_active = True
        state, fresh = self.tracker.update_gate4(values, now_s=now)
        if state is None:
            quaternion = np.asarray(values[6:10], dtype=np.float64)
            yaw = float(mpc.quaternion_to_euler(quaternion)[2])
            self.cached_action = (0.25, 0.0, 0.0, tail._clamp(yaw / math.pi))
            self.failsafe_plans += 1
            return list(self.cached_action)

        if self.yaw_target_rad is None:
            current_yaw = float(state.euler_rpy_rad[2])
            yaw_error = tail._clamp(float(values[14])) * (math.pi / 4.0)
            yaw_step = mpc.clamp(
                yaw_error, -MAXIMUM_YAW_STEP_RAD, MAXIMUM_YAW_STEP_RAD
            )
            self.yaw_target_rad = mpc.wrap_angle(current_yaw - yaw_step)

        time_since_plan_s = (
            math.inf if self.last_plan_s is None else now - self.last_plan_s
        )
        replan = (
            self.cached_action is None
            or self.last_plan_s is None
            or time_since_plan_s >= REPLAN_MAX_INTERVAL_S
            or (fresh and time_since_plan_s >= REPLAN_MIN_INTERVAL_S)
        )
        if replan:
            plan = self.optimizer.plan(state, yaw_target_rad=self.yaw_target_rad)
            self.last_plan = plan
            self.last_plan_s = now
            self.cached_action = plan.action
            if plan.accepted:
                self.optimized_plans += 1
            else:
                self.failsafe_plans += 1
        else:
            self.cached_action_calls += 1
        assert self.cached_action is not None
        return [tail._clamp(value) for value in self.cached_action]

    def snapshot(self) -> dict:
        return {
            "gate4_phase_active": self.gate4_phase_active,
            "yaw_target_rad": self.yaw_target_rad,
            "cached_action": self.cached_action,
            "last_plan_s": self.last_plan_s,
            "last_plan": None
            if self.last_plan is None
            else dataclasses.asdict(self.last_plan),
            "inference_calls": self.inference_calls,
            "gate4_calls": self.gate4_calls,
            "optimized_plans": self.optimized_plans,
            "failsafe_plans": self.failsafe_plans,
            "cached_action_calls": self.cached_action_calls,
            "tracker": self.tracker.snapshot(),
            "optimizer_config": dataclasses.asdict(self.optimizer.config),
            "contract": {
                "gate4_index": GATE4_INDEX,
                "gates_1_3_parent_invariant": True,
                "gates_5_6_parent_invariant": True,
                "execute_first_control_only": True,
                "maximum_yaw_step_rad": MAXIMUM_YAW_STEP_RAD,
                "replan_min_interval_s": REPLAN_MIN_INTERVAL_S,
                "replan_max_interval_s": REPLAN_MAX_INTERVAL_S,
                "logical_policy_state_hz": POLICY_LOGICAL_HZ,
                "runner_policy_state_hz": 0.0,
                "model_sha256": EXPECTED_MODEL_SHA256,
                "runtime_privileged_state": False,
            },
        }


_CONTROLLER: Gate4MPCController | None = None
_CONTROLLER_MODEL_PATH: Path | None = None


def _controller() -> Gate4MPCController:
    global _CONTROLLER, _CONTROLLER_MODEL_PATH
    path = _required_model_path()
    if _CONTROLLER is None or _CONTROLLER_MODEL_PATH != path:
        _CONTROLLER = Gate4MPCController(mpc.VisualDynamicsEnsemble.load(path))
        _CONTROLLER_MODEL_PATH = path
    return _CONTROLLER


def reset() -> None:
    parent.reset()
    _controller().reset()


def infer(observation: Sequence[float]) -> list[float]:
    # The parent always advances first, including at Gate 4, so recurrent state
    # and the Gates-5/6 handoff retain their established update schedule.
    base_actions = parent.infer(observation)
    return _controller().apply(observation, base_actions)


def controller_snapshot() -> dict:
    return {
        "n232_gate4_visual_mpc": _controller().snapshot(),
        "parent": parent.controller_snapshot(),
    }
