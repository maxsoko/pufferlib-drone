#!/usr/bin/env python3
"""N240: admit a continuously associated family when it enters 16 metres.

N239's official batch never exercised its learned lateral MPC because trust was
latched only on initial acquisition or discrete reacquisition.  All three live
traces contained already-associated families that later reached the unchanged
16 m boundary.  N240 changes only that state transition: any *accepted*
coherent association sample may latch near-family trust at the same boundary.
Rejected raw poses remain inert and every control/model contract is inherited.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import gate4_identity_body_mpc as body_mpc
import policy_callable_gate4_identity_body_mpc_n239 as n239


EXPECTED_N239_SHA256 = (
    "716352e7b2ec603e0316e59f0d97cc6fe96ca5bdde4f5f7923b0be2fb6b75f4c"
)
POLICY_STATE_HZ = n239.POLICY_STATE_HZ
TRUSTED_FAMILY_MAXIMUM_FORWARD_M = (
    n239.TRUSTED_FAMILY_MAXIMUM_ACQUISITION_FORWARD_M
)
# The float observation/tanh inverse reconstructs a nominal 16.000000 m sample
# as 16.000003 m in one official trace.  One millimetre is an explicit encoder
# boundary tolerance, not a meaningful extension of the control envelope.
FORWARD_BOUNDARY_TOLERANCE_M = 1e-3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_N239_PATH = Path(n239.__file__).resolve()
if _sha256(_N239_PATH) != EXPECTED_N239_SHA256:
    raise RuntimeError("N239 policy source hash mismatch")


class Gate4ContinuousNearMPCController(n239.Gate4IdentityBodyMPCController):
    """Correct N239's admission transition without changing its controller."""

    def reset(self) -> None:
        super().reset()
        self.continuous_near_family_triggers = 0

    def _accept(
        self,
        pose: np.ndarray,
        values: np.ndarray,
        *,
        now_s: float,
        reacquired: bool,
    ) -> None:
        was_trusted = self.trusted_near_family
        super()._accept(
            pose,
            values,
            now_s=now_s,
            reacquired=reacquired,
        )
        if (
            not was_trusted
            and not self.trusted_near_family
            and float(pose[0])
            <= TRUSTED_FAMILY_MAXIMUM_FORWARD_M + FORWARD_BOUNDARY_TOLERANCE_M
        ):
            self.trusted_near_family = True
            self.trusted_family_triggers += 1
            self.continuous_near_family_triggers += 1

    def snapshot(self) -> dict:
        snapshot = super().snapshot()
        snapshot["continuous_near_family_admission"] = {
            "continuous_near_family_triggers": (
                self.continuous_near_family_triggers
            ),
            "trusted_family_maximum_forward_m": (
                TRUSTED_FAMILY_MAXIMUM_FORWARD_M
            ),
            "forward_boundary_tolerance_m": FORWARD_BOUNDARY_TOLERANCE_M,
            "contract": {
                "trust_source_is_accepted_association_only": True,
                "continuous_association_may_latch_trust": True,
                "nominal_threshold_changed_from_n239": False,
                "encoder_boundary_tolerance_is_explicit": True,
                "rejected_pose_action_authority": False,
                "model_changed_from_n239": False,
                "control_law_changed_from_n239": False,
                "pitch_thrust_yaw_changed_from_n239": False,
                "runtime_privileged_state": False,
            },
        }
        return snapshot


_CONTROLLER: Gate4ContinuousNearMPCController | None = None
_CONTROLLER_MODEL_PATH: Path | None = None
_INFERENCE_CALLS = 0


def _controller() -> Gate4ContinuousNearMPCController:
    global _CONTROLLER, _CONTROLLER_MODEL_PATH
    if _CONTROLLER is None:
        path = n239._required_model_path()
        ensemble = body_mpc.LateralDynamicsEnsemble.load(path)
        _CONTROLLER = Gate4ContinuousNearMPCController(ensemble)
        _CONTROLLER_MODEL_PATH = path
    return _CONTROLLER


def reset() -> None:
    global _INFERENCE_CALLS
    n239.n238.n236.n235.n234.n233.n232.parent.reset()
    _controller().reset()
    _INFERENCE_CALLS = 0


def infer(observation: Sequence[float]) -> list[float]:
    global _INFERENCE_CALLS
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    parent_actions = n239.n238.n236.n235.n234.n233.n232.parent.infer(values)
    logical_time_s = _INFERENCE_CALLS / POLICY_STATE_HZ
    _INFERENCE_CALLS += 1
    return _controller().apply(values, parent_actions, now_s=logical_time_s)


def controller_snapshot() -> dict:
    return {
        "n240_gate4_continuous_near_mpc": _controller().snapshot(),
        "parent": n239.n238.n236.n235.n234.n233.n232.parent.controller_snapshot(),
    }
