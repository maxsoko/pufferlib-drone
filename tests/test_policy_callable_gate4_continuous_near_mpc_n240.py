from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gate4_identity_body_mpc as body_mpc
import policy_callable_gate4_continuous_near_mpc_n240 as n240


MODEL = ROOT / "logs/sitl/n239_gate4_identity_body_dynamics_fit_20260719.json"


def _observation(gate: int, pose=None):
    values = np.zeros(32, dtype=np.float64)
    values[6] = 1.0
    values[23] = gate / 6.0
    if pose is not None:
        values[10] = 1.0
        values[11] = math.tanh(pose[0] / 10.0)
        values[12] = math.tanh(pose[1] / 5.0)
        values[13] = math.tanh(pose[2] / 5.0)
        values[14] = math.atan2(pose[1], pose[0]) / (math.pi / 4.0)
    return values


def _controller() -> n240.Gate4ContinuousNearMPCController:
    return n240.Gate4ContinuousNearMPCController(
        body_mpc.LateralDynamicsEnsemble.load(MODEL)
    )


def _apply(controller, pose, now_s):
    return controller.apply(
        _observation(3, pose), [0.0, 0.0, 0.0, 0.0], now_s=now_s
    )


def test_far_family_remains_quarantined() -> None:
    controller = _controller()
    output = None
    for index, pose in enumerate(
        ((35.0, 3.0, 4.0), (34.0, 2.8, 3.8), (33.0, 2.6, 3.6))
    ):
        output = _apply(controller, pose, 1.0 + index * 0.1)
    assert output[:3] == [n240.n239.n238.BRAKE_PITCH_NORM, 0.0, 0.0]
    assert not controller.trusted_near_family
    assert controller.continuous_near_family_triggers == 0
    assert controller.optimized_lateral_plans == 0


def test_continuously_associated_family_latches_at_unchanged_boundary() -> None:
    controller = _controller()
    poses = [
        (forward, 0.15 * (forward - 16.0), 0.2 * (forward - 16.0))
        for forward in np.linspace(35.0, 16.0, 20)
    ]
    poses.append((15.9, -0.1, 0.1))
    output = None
    for index, pose in enumerate(poses):
        output = _apply(controller, pose, 1.0 + index * 0.1)
    assert controller.trusted_near_family
    assert controller.trusted_family_triggers == 1
    assert controller.continuous_near_family_triggers == 1
    assert controller.closer_reacquisitions == 0
    assert controller.optimized_lateral_plans >= 1
    assert output[1] != 0.0
    admission = controller.snapshot()["continuous_near_family_admission"]
    assert admission["contract"]["nominal_threshold_changed_from_n239"] is False
    assert admission["forward_boundary_tolerance_m"] == 1e-3
    assert admission["contract"]["rejected_pose_action_authority"] is False


def test_rejected_alias_remains_inert_after_continuous_unlock() -> None:
    controller = _controller()
    poses = [(35.0 - index, 2.0, 2.0) for index in range(20)]
    poses.append((15.9, 1.9, 1.9))
    trusted = None
    for index, pose in enumerate(poses):
        trusted = _apply(controller, pose, 1.0 + index * 0.1)
    assert controller.trusted_near_family
    plan_count = controller.optimized_lateral_plans
    alias = _apply(controller, (36.0, 12.0, 9.0), 2.0)
    assert alias == trusted
    assert controller.optimized_lateral_plans == plan_count


def test_non_gate4_parent_actions_remain_exact() -> None:
    controller = _controller()
    parent = [0.123, -0.456, 0.789, -0.234]
    for gate in (0, 1, 2, 4, 5):
        assert controller.apply(
            _observation(gate), parent, now_s=gate / 10.0
        ) == parent
