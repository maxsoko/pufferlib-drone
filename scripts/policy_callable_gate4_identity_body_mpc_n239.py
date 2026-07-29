#!/usr/bin/env python3
"""N239: identity-associated, data-trained robust lateral Gate-4 MPC.

N238 proved that its persistent observable association removed rejected-alias
action authority, but its hand-tuned projected roll reached the Gate-4 frame
with +5 to +7 m/s of rightward image-relative motion.  N239 retains N238's
association, pitch, thrust, yaw, dropout, and commit rules verbatim.  Only its
roll channel is replaced by deterministic receding-horizon optimization over
six leave-one-official-run body-relative dynamics models.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import gate4_identity_body_mpc as body_mpc
import policy_callable_gate4_identity_locked_intercept_n238 as n238


EXPECTED_N238_SHA256 = (
    "2e2f05b25907a14ba062f2a140c1068fbc625693302165b97cf811e00d43b8fc"
)
EXPECTED_BODY_MPC_SHA256 = (
    "1059acff7135bbdd7bb20fc9630891d28cda7794630a3a51a590fc014d96338b"
)
EXPECTED_MODEL_SHA256 = (
    "a40e9279bc2d34f69a2296350971ee27eb1393f64e5527b37f2bdc1a545677df"
)
MODEL_ENVIRONMENT_VARIABLE = "PUFFER_POLICY_GATE4_IDENTITY_DYNAMICS_PATH"
POLICY_STATE_HZ = 60.0
TRUSTED_FAMILY_MAXIMUM_ACQUISITION_FORWARD_M = 16.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_N238_PATH = Path(n238.__file__).resolve()
if _sha256(_N238_PATH) != EXPECTED_N238_SHA256:
    raise RuntimeError("N238 policy source hash mismatch")
_BODY_MPC_PATH = Path(body_mpc.__file__).resolve()
if _sha256(_BODY_MPC_PATH) != EXPECTED_BODY_MPC_SHA256:
    raise RuntimeError("N239 body MPC source hash mismatch")


def _required_model_path() -> Path:
    value = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    if not value:
        raise RuntimeError(f"{MODEL_ENVIRONMENT_VARIABLE} is required")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "N239 identity dynamics hash mismatch: "
            f"expected {EXPECTED_MODEL_SHA256}, got {actual}"
        )
    return path


class Gate4IdentityBodyMPCController(
    n238.Gate4IdentityLockedInterceptController
):
    """Use N238 state ownership and replace only fresh associated roll."""

    def __init__(self, ensemble: body_mpc.LateralDynamicsEnsemble) -> None:
        self.optimizer = body_mpc.Gate4IdentityLateralMPC(ensemble)
        super().__init__()

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "optimizer"):
            self.optimizer.reset()
        self.last_planned_accepted_fresh = 0
        self.cached_mpc_roll: float | None = None
        self.last_lateral_plan: body_mpc.LateralPlan | None = None
        self.optimized_lateral_plans = 0
        self.robust_centered_plans = 0
        self.robust_uncentered_plans = 0
        self.cached_lateral_action_calls = 0
        self.trusted_near_family = False
        self.trusted_family_triggers = 0
        self.quarantined_associated_calls = 0

    def _accept(
        self,
        pose: np.ndarray,
        values: np.ndarray,
        *,
        now_s: float,
        reacquired: bool,
    ) -> None:
        initial = self.associated_pose is None
        super()._accept(
            pose,
            values,
            now_s=now_s,
            reacquired=reacquired,
        )
        if (
            not self.trusted_near_family
            and (initial or reacquired)
            and float(pose[0])
            <= TRUSTED_FAMILY_MAXIMUM_ACQUISITION_FORWARD_M
        ):
            self.trusted_near_family = True
            self.trusted_family_triggers += 1

    def _associated_action(self, values: np.ndarray, *, now_s: float) -> list[float]:
        if not self.trusted_near_family:
            self.quarantined_associated_calls += 1
            self.cached_mpc_roll = None
            return self._level_brake(values)
        action = super()._associated_action(values, now_s=now_s)
        assert self.associated_pose is not None
        assert self.associated_last_s is not None
        age_s = max(0.0, now_s - self.associated_last_s)
        if age_s > n238.ASSOCIATED_HOLD_S:
            self.cached_mpc_roll = None
            return action

        if self.accepted_fresh != self.last_planned_accepted_fresh:
            forward_m, right_m = (
                float(self.associated_pose[index]) for index in (0, 1)
            )
            forward_rate_m_s, right_rate_m_s = (
                float(self.associated_rate_m_s[index]) for index in (0, 1)
            )
            plan = self.optimizer.plan(
                forward_m=forward_m,
                forward_rate_m_s=forward_rate_m_s,
                right_m=right_m,
                right_rate_m_s=right_rate_m_s,
            )
            self.last_lateral_plan = plan
            self.cached_mpc_roll = plan.roll_norm
            self.last_planned_accepted_fresh = self.accepted_fresh
            self.optimized_lateral_plans += 1
            if plan.accepted:
                self.robust_centered_plans += 1
            else:
                self.robust_uncentered_plans += 1
        else:
            self.cached_lateral_action_calls += 1

        if self.cached_mpc_roll is not None:
            action[1] = body_mpc.clamp(self.cached_mpc_roll)
        return action

    def snapshot(self) -> dict:
        snapshot = super().snapshot()
        snapshot["identity_body_lateral_mpc"] = {
            "last_planned_accepted_fresh": self.last_planned_accepted_fresh,
            "cached_mpc_roll": self.cached_mpc_roll,
            "last_lateral_plan": (
                None
                if self.last_lateral_plan is None
                else dataclasses.asdict(self.last_lateral_plan)
            ),
            "optimized_lateral_plans": self.optimized_lateral_plans,
            "robust_centered_plans": self.robust_centered_plans,
            "robust_uncentered_plans": self.robust_uncentered_plans,
            "cached_lateral_action_calls": self.cached_lateral_action_calls,
            "trusted_near_family": self.trusted_near_family,
            "trusted_family_triggers": self.trusted_family_triggers,
            "quarantined_associated_calls": self.quarantined_associated_calls,
            "optimizer_config": dataclasses.asdict(self.optimizer.config),
            "model": {
                "source_path": self.optimizer.ensemble.source_path,
                "source_sha256": self.optimizer.ensemble.source_sha256,
                "members": self.optimizer.ensemble.members,
                "minimum_roll_gain_m_s2": float(
                    np.min(self.optimizer.ensemble.roll_gain_m_s2)
                ),
                "maximum_roll_gain_m_s2": float(
                    np.max(self.optimizer.ensemble.roll_gain_m_s2)
                ),
            },
            "contract": {
                "trained_official_associated_transitions": 113,
                "whole_run_fold_members": self.optimizer.ensemble.members,
                "optimized_axis": body_mpc.DEPLOYABLE_AXIS,
                "pitch_thrust_yaw_inherited_exactly_from_n238": True,
                "far_associated_families_have_action_authority": False,
                "trusted_family_maximum_acquisition_forward_m": (
                    TRUSTED_FAMILY_MAXIMUM_ACQUISITION_FORWARD_M
                ),
                "rejected_pose_action_authority": False,
                "execute_first_control_only": True,
                "deterministic_optimizer": True,
                "runtime_privileged_state": False,
            },
        }
        return snapshot


_CONTROLLER: Gate4IdentityBodyMPCController | None = None
_CONTROLLER_MODEL_PATH: Path | None = None
_INFERENCE_CALLS = 0


def _controller() -> Gate4IdentityBodyMPCController:
    global _CONTROLLER, _CONTROLLER_MODEL_PATH
    # Deployment fixes this environment variable before importing the policy.
    # Hashing an UNC model file at every 60 Hz state step creates a catch-up
    # feedback loop, so validate and load exactly once per policy process.
    if _CONTROLLER is None:
        path = _required_model_path()
        ensemble = body_mpc.LateralDynamicsEnsemble.load(path)
        _CONTROLLER = Gate4IdentityBodyMPCController(ensemble)
        _CONTROLLER_MODEL_PATH = path
    return _CONTROLLER


def reset() -> None:
    global _INFERENCE_CALLS
    n238.n236.n235.n234.n233.n232.parent.reset()
    _controller().reset()
    _INFERENCE_CALLS = 0


def infer(observation: Sequence[float]) -> list[float]:
    global _INFERENCE_CALLS
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    parent_actions = n238.n236.n235.n234.n233.n232.parent.infer(values)
    logical_time_s = _INFERENCE_CALLS / POLICY_STATE_HZ
    _INFERENCE_CALLS += 1
    return _controller().apply(
        values, parent_actions, now_s=logical_time_s
    )


def controller_snapshot() -> dict:
    return {
        "n239_gate4_identity_body_lateral_mpc": _controller().snapshot(),
        "parent": n238.n236.n235.n234.n233.n232.parent.controller_snapshot(),
    }
