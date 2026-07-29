#!/usr/bin/env python3
"""N241: bounded Gate-4 roll excitation for causal system identification.

The nine-run passive fit fails its fold-consistency gate because roll action
and ambient lateral drift remain confounded.  N241 is not a lap candidate.  It
uses N238's persistent accepted association and applies a balanced, two-second
roll sequence only while that association is fresh and 20-40 m forward.  Pitch,
thrust, and yaw stay at the level/brake fallback.  Missing, stale, near, far, or
rejected observations have zero excitation authority.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import policy_callable_gate4_identity_locked_intercept_n238 as n238


EXPECTED_N238_SHA256 = (
    "2e2f05b25907a14ba062f2a140c1068fbc625693302165b97cf811e00d43b8fc"
)
POLICY_STATE_HZ = 60.0
PROBE_MINIMUM_FORWARD_M = 20.0
PROBE_MAXIMUM_FORWARD_M = 40.0
PULSE_DURATION_S = 0.25
ROLL_SEQUENCE = (-0.35, 0.35, -0.35, 0.35, -0.20, 0.20, -0.20, 0.20)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_N238_PATH = Path(n238.__file__).resolve()
if _sha256(_N238_PATH) != EXPECTED_N238_SHA256:
    raise RuntimeError("N238 policy source hash mismatch")


class Gate4RollProbeController(n238.Gate4IdentityLockedInterceptController):
    """Inject balanced roll only from a fresh accepted far association."""

    def reset(self) -> None:
        super().reset()
        self.probe_start_s: float | None = None
        self.probe_action_calls = 0
        self.probe_completed = False
        self.probe_aborted_near = False
        self.probe_roll_counts = {str(value): 0 for value in ROLL_SEQUENCE}

    def _accept(
        self,
        pose: np.ndarray,
        values: np.ndarray,
        *,
        now_s: float,
        reacquired: bool,
    ) -> None:
        super()._accept(
            pose,
            values,
            now_s=now_s,
            reacquired=reacquired,
        )
        if (
            self.probe_start_s is not None
            and not self.probe_completed
            and float(pose[0]) < PROBE_MINIMUM_FORWARD_M
        ):
            self.probe_aborted_near = True

    def _associated_action(self, values: np.ndarray, *, now_s: float) -> list[float]:
        assert self.associated_pose is not None
        assert self.associated_last_s is not None
        age_s = max(0.0, now_s - self.associated_last_s)
        forward_m = float(self.associated_pose[0])
        if self.probe_aborted_near:
            return self._level_brake(values)
        if age_s > n238.ASSOCIATED_HOLD_S:
            return self._level_brake(values)
        if forward_m < PROBE_MINIMUM_FORWARD_M:
            if self.probe_start_s is not None and not self.probe_completed:
                self.probe_aborted_near = True
            return self._level_brake(values)
        if forward_m > PROBE_MAXIMUM_FORWARD_M:
            return self._level_brake(values)
        if self.probe_completed:
            return self._level_brake(values)
        if self.probe_start_s is None:
            self.probe_start_s = now_s
        elapsed_s = max(0.0, now_s - self.probe_start_s)
        phase = int(elapsed_s / PULSE_DURATION_S)
        if phase >= len(ROLL_SEQUENCE):
            self.probe_completed = True
            return self._level_brake(values)

        action = self._level_brake(values)
        roll = float(ROLL_SEQUENCE[phase])
        action[1] = roll
        self.probe_action_calls += 1
        self.probe_roll_counts[str(roll)] += 1
        self.associated_control_calls += 1
        return action

    def snapshot(self) -> dict:
        snapshot = super().snapshot()
        snapshot["gate4_roll_identification_probe"] = {
            "probe_start_s": self.probe_start_s,
            "probe_action_calls": self.probe_action_calls,
            "probe_completed": self.probe_completed,
            "probe_aborted_near": self.probe_aborted_near,
            "probe_roll_counts": self.probe_roll_counts,
            "roll_sequence": list(ROLL_SEQUENCE),
            "pulse_duration_s": PULSE_DURATION_S,
            "contract": {
                "lap_candidate": False,
                "accepted_association_required": True,
                "maximum_association_age_s": n238.ASSOCIATED_HOLD_S,
                "probe_minimum_forward_m": PROBE_MINIMUM_FORWARD_M,
                "probe_maximum_forward_m": PROBE_MAXIMUM_FORWARD_M,
                "balanced_roll_sequence": abs(sum(ROLL_SEQUENCE)) < 1e-12,
                "pitch_thrust_yaw_are_level_brake": True,
                "rejected_pose_action_authority": False,
                "runtime_privileged_state": False,
                "gates_1_3_parent_invariant": True,
                "gates_5_6_parent_invariant": True,
            },
        }
        return snapshot


_CONTROLLER = Gate4RollProbeController()
_INFERENCE_CALLS = 0


def reset() -> None:
    global _INFERENCE_CALLS
    n238.n236.n235.n234.n233.n232.parent.reset()
    _CONTROLLER.reset()
    _INFERENCE_CALLS = 0


def infer(observation: Sequence[float]) -> list[float]:
    global _INFERENCE_CALLS
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (32,):
        raise ValueError(f"expected observation shape (32,), got {values.shape}")
    parent_actions = n238.n236.n235.n234.n233.n232.parent.infer(values)
    logical_time_s = _INFERENCE_CALLS / POLICY_STATE_HZ
    _INFERENCE_CALLS += 1
    return _CONTROLLER.apply(values, parent_actions, now_s=logical_time_s)


def controller_snapshot() -> dict:
    return {
        "n241_gate4_roll_identification_probe": _CONTROLLER.snapshot(),
        "parent": n238.n236.n235.n234.n233.n232.parent.controller_snapshot(),
    }
