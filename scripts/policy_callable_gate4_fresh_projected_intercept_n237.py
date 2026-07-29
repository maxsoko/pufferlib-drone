#!/usr/bin/env python3
"""N237: action-propagated observer plus fresh projected Gate-4 intercept.

N236 proved that braking before centering avoids the immediate N235 impact,
but its low-gain raw fallback could not remove the measured lateral/vertical
miss before the Gate-4 plane.  The existing bounded optimizer already returns
a velocity-aware projected intercept action when a fresh plan is rejected for
state/model disagreement.  N237 prevents aperture/range jumps from becoming
fictional velocity by propagating the reliable Gate-3 entry velocity through
the fitted plant using actual action history. Camera poses correct position,
not velocity. The projected action is executed only on the fresh frame that
produced it. Missing, repeated, stale, or incoherent measurements retain
N236's staged brake/search behavior and the 0.30 s hard track reset.
"""

from __future__ import annotations

import dataclasses
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
MAXIMUM_INITIAL_RANGE_M = 40.0
MAXIMUM_OBSERVER_STEP_S = 0.10
OBSERVER_UNCERTAINTY_LIMIT_M = 2.0
PROJECTED_INTERCEPT_REASONS = frozenset(
    {"state_uncertain", "prediction_uncertain"}
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_N236_PATH = Path(n236.__file__).resolve()
if _sha256(_N236_PATH) != EXPECTED_N236_SHA256:
    raise RuntimeError("N236 policy source hash mismatch")


class Gate4FreshProjectedInterceptController(n236.Gate4StagedMPCController):
    def __init__(
        self, ensemble: n236.n235.n234.n233.n232.mpc.VisualDynamicsEnsemble
    ) -> None:
        self.ensemble = ensemble
        tracker_config = dataclasses.replace(
            n236.n235.n234.n233.n232.mpc.TrackerConfig(),
            maximum_initial_range_m=MAXIMUM_INITIAL_RANGE_M,
        )
        self.tracker = n236.n235.n234.n233.n232.mpc.Gate4VisualStateTracker(
            tracker_config
        )
        optimizer_config = dataclasses.replace(
            n236.n235.n234.n233.n232.mpc.OptimizerConfig(),
            uncertainty_limit_m=OBSERVER_UNCERTAINTY_LIMIT_M,
        )
        self.optimizer = (
            n236.n235.n234.n233.n232.mpc.Gate4RecedingHorizonOptimizer(
                ensemble, optimizer_config
            )
        )
        self.mean_thrust_gain = np.mean(ensemble.thrust_gain, axis=0)
        self.mean_linear_drag_per_s = np.mean(
            ensemble.linear_drag_per_s, axis=0
        )
        self.mean_acceleration_bias_m_s2 = np.mean(
            ensemble.acceleration_bias_m_s2, axis=0
        )
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.fresh_projected_intercept_plans = 0
        self.rejected_plan_neutralizations = 0
        self.observer_velocity_world_m_s: np.ndarray | None = None
        self.observer_last_s: float | None = None
        self.observer_last_action = (0.0, 0.0, 0.0, 0.0)
        self.observer_propagations = 0
        self.observer_velocity_restorations = 0
        self.last_visual_state = None

    def _leave_gate4(self) -> None:
        was_active = self.gate4_phase_active
        super()._leave_gate4()
        if was_active:
            self.observer_velocity_world_m_s = None
            self.observer_last_s = None
            self.observer_last_action = (0.0, 0.0, 0.0, 0.0)
            self.last_visual_state = None

    def _propagate_observer(self, values: np.ndarray, *, now_s: float) -> None:
        if self.observer_velocity_world_m_s is None:
            carried = self.tracker.carried_velocity_world_m_s
            if carried is not None:
                self.observer_velocity_world_m_s = carried.copy()
        if self.observer_last_s is None:
            self.observer_last_s = float(now_s)
            return
        dt = n236.n235.n234.n233.n232.mpc.clamp(
            float(now_s) - self.observer_last_s,
            0.0,
            MAXIMUM_OBSERVER_STEP_S,
        )
        self.observer_last_s = float(now_s)
        if self.observer_velocity_world_m_s is None or dt <= 0.0:
            return
        quaternion = np.asarray(values[6:10], dtype=np.float64)
        up = n236.n235.n234.n233.n232.mpc.quaternion_rotate(
            quaternion, (0.0, 0.0, 1.0)
        )
        thrust_axis = up * float(
            n236.n235.n234.n233.n232.mpc.command_thrust(
                self.observer_last_action[2]
            )
        )
        thrust_axis[0] *= -1.0
        acceleration = (
            self.mean_thrust_gain * thrust_axis
            - self.mean_linear_drag_per_s * self.observer_velocity_world_m_s
            + self.mean_acceleration_bias_m_s2
        )
        self.observer_velocity_world_m_s += acceleration * dt
        self.observer_propagations += 1
        if self.tracker.velocity_world_m_s is not None:
            self.tracker.velocity_world_m_s = (
                self.observer_velocity_world_m_s.copy()
            )

    def _restore_observer_velocity(self, state):
        if self.observer_velocity_world_m_s is None:
            if self.tracker.velocity_world_m_s is not None:
                self.observer_velocity_world_m_s = (
                    self.tracker.velocity_world_m_s.copy()
                )
            else:
                return state
        if self.tracker.velocity_world_m_s is not None:
            self.tracker.velocity_world_m_s = (
                self.observer_velocity_world_m_s.copy()
            )
        if state is not None:
            state.velocity_world_m_s = self.observer_velocity_world_m_s.copy()
            self.observer_velocity_restorations += 1
        return state

    def _reset_stale_track(self, _values: np.ndarray) -> None:
        velocity = (
            None
            if self.observer_velocity_world_m_s is None
            else self.observer_velocity_world_m_s.copy()
        )
        self.tracker.reset()
        self.tracker.carried_velocity_world_m_s = velocity
        self.optimizer.reset()
        self.yaw_target_rad = None
        self.cached_action = None
        self.last_plan_s = None
        self.last_plan = None
        self.coherent_fresh_samples = 0
        self.track_resets += 1

    def _publish(self, action: Sequence[float]) -> list[float]:
        output = [
            n236.n235.n234.n233.n232.tail._clamp(value) for value in action
        ]
        self.observer_last_action = tuple(output)
        return output

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
        gate = n236.n235.n234.n233.n232.tail._gate_index(values)
        now = float(now_s)
        self.inference_calls += 1

        if gate == n236.n235.n234.n233.n232.GATE3_INDEX:
            self.tracker.observe_prefix(values, gate_index=gate)
            self._leave_gate4()
            carried = self.tracker.carried_velocity_world_m_s
            self.observer_velocity_world_m_s = (
                None if carried is None else carried.copy()
            )
            self.observer_last_s = now
            return self._publish(actions)
        if gate != n236.n235.n234.n233.n232.GATE4_INDEX:
            self._leave_gate4()
            return actions

        self.control_now_s = now
        self.gate4_calls += 1
        self.gate4_phase_active = True
        self._propagate_observer(values, now_s=now)
        rejected_before = self.tracker.rejected_samples
        state, fresh = self.tracker.update_gate4(values, now_s=now)
        state = self._restore_observer_velocity(state)
        if (
            state is not None
            and state.measurement_age_s > n236.n235.n234.TRACK_STALE_RESET_S
        ):
            self._reset_stale_track(values)
            rejected_before = self.tracker.rejected_samples
            state, fresh = self.tracker.update_gate4(values, now_s=now)
            state = self._restore_observer_velocity(state)
        self.last_visual_state = None if state is None else state.copy()

        if fresh:
            self.coherent_fresh_samples += 1
        elif self.tracker.rejected_samples > rejected_before:
            self.coherent_fresh_samples = 0

        if (
            state is None
            or self.coherent_fresh_samples
            < n236.n235.n234.REQUIRED_COHERENT_FRESH_SAMPLES
        ):
            self.acquisition_wait_calls += 1
            self.cached_action = tuple(self._neutral_action(values))
            return self._publish(self.cached_action)

        if self.yaw_target_rad is None:
            current_yaw = float(state.euler_rpy_rad[2])
            yaw_error = n236.n235.n234.n233.n232.tail._clamp(
                float(values[14])
            ) * (math.pi / 4.0)
            yaw_step = n236.n235.n234.n233.n232.mpc.clamp(
                yaw_error,
                -n236.n235.n234.n233.n232.MAXIMUM_YAW_STEP_RAD,
                n236.n235.n234.n233.n232.MAXIMUM_YAW_STEP_RAD,
            )
            self.yaw_target_rad = n236.n235.n234.n233.n232.mpc.wrap_angle(
                current_yaw - yaw_step
            )

        time_since_plan_s = (
            math.inf if self.last_plan_s is None else now - self.last_plan_s
        )
        replan = (
            self.cached_action is None
            or self.last_plan_s is None
            or time_since_plan_s
            >= n236.n235.n234.n233.n232.REPLAN_MAX_INTERVAL_S
            or (
                fresh
                and time_since_plan_s
                >= n236.n235.n234.n233.n232.REPLAN_MIN_INTERVAL_S
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
                use_projected_intercept = (
                    fresh
                    and state.measurement_age_s
                    <= n236.n235.n234.TRACK_STALE_RESET_S
                    and plan.reason in PROJECTED_INTERCEPT_REASONS
                )
                if use_projected_intercept:
                    self.cached_action = plan.action
                    self.fresh_projected_intercept_plans += 1
                else:
                    self.cached_action = tuple(self._neutral_action(values))
                    self.rejected_plan_neutralizations += 1
                self.rejected_plans += 1
                self.plan_rejection_counts[plan.reason] += 1
        else:
            self.cached_action_calls += 1
        assert self.cached_action is not None
        return self._publish(self.cached_action)

    def snapshot(self) -> dict:
        snapshot = super().snapshot()
        snapshot["fresh_projected_intercept"] = {
            "plans": self.fresh_projected_intercept_plans,
            "rejected_plan_neutralizations": self.rejected_plan_neutralizations,
            "allowed_rejection_reasons": sorted(PROJECTED_INTERCEPT_REASONS),
            "contract": {
                "requires_fresh_measurement": True,
                "requires_coherent_track": True,
                "maximum_measurement_age_s": n236.n235.n234.TRACK_STALE_RESET_S,
                "n236_missing_stale_and_incoherent_behavior_unchanged": True,
                "gates_1_3_parent_invariant": True,
                "gates_5_6_parent_invariant": True,
                "runtime_privileged_state": False,
            },
        }
        snapshot["action_propagated_observer"] = {
            "velocity_world_m_s": (
                None
                if self.observer_velocity_world_m_s is None
                else self.observer_velocity_world_m_s.tolist()
            ),
            "last_s": self.observer_last_s,
            "last_action": self.observer_last_action,
            "propagations": self.observer_propagations,
            "velocity_restorations": self.observer_velocity_restorations,
            "mean_thrust_gain": self.mean_thrust_gain.tolist(),
            "mean_linear_drag_per_s": self.mean_linear_drag_per_s.tolist(),
            "mean_acceleration_bias_m_s2": (
                self.mean_acceleration_bias_m_s2.tolist()
            ),
            "contract": {
                "maximum_initial_range_m": MAXIMUM_INITIAL_RANGE_M,
                "maximum_observer_step_s": MAXIMUM_OBSERVER_STEP_S,
                "uncertainty_limit_m": OBSERVER_UNCERTAINTY_LIMIT_M,
                "gate3_entry_velocity_prior": True,
                "action_history_propagation": True,
                "camera_corrects_position_not_velocity": True,
                "runtime_privileged_state": False,
            },
        }
        return snapshot


_CONTROLLER: Gate4FreshProjectedInterceptController | None = None
_INFERENCE_CALLS = 0


def _controller() -> Gate4FreshProjectedInterceptController:
    global _CONTROLLER
    if _CONTROLLER is None:
        verified = n236.n235.n234.n233._load_verified_controller()
        _CONTROLLER = Gate4FreshProjectedInterceptController(verified.ensemble)
    return _CONTROLLER


def reset() -> None:
    global _INFERENCE_CALLS
    n236.n235.n234.n233.n232.parent.reset()
    _controller().reset()
    _INFERENCE_CALLS = 0


def infer(observation: Sequence[float]) -> list[float]:
    global _INFERENCE_CALLS
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    parent_actions = n236.n235.n234.n233.n232.parent.infer(values)
    logical_time_s = _INFERENCE_CALLS / POLICY_STATE_HZ
    _INFERENCE_CALLS += 1
    return _controller().apply(values, parent_actions, now_s=logical_time_s)


def controller_snapshot() -> dict:
    return {
        "n237_gate4_fresh_projected_intercept": _controller().snapshot(),
        "n233_model_validation": {
            "model_validations": n236.n235.n234.n233._MODEL_VALIDATIONS,
            "model_path": (
                None
                if n236.n235.n234.n233._CONTROLLER_MODEL_PATH is None
                else str(n236.n235.n234.n233._CONTROLLER_MODEL_PATH)
            ),
        },
        "parent": n236.n235.n234.n233.n232.parent.controller_snapshot(),
    }
