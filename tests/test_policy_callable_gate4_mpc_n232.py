from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gate4_receding_horizon as mpc
import policy_callable_gate4_mpc_n232 as n232


def _ensemble() -> mpc.VisualDynamicsEnsemble:
    return mpc.VisualDynamicsEnsemble(
        thrust_gain=np.asarray([[60.0, 15.0, 40.0], [70.0, 18.0, 45.0]]),
        linear_drag_per_s=np.asarray([[0.0, 0.0, 0.75], [0.0, 0.2, 0.85]]),
        acceleration_bias_m_s2=np.asarray([[0.0, 0.0, -11.0], [0.0, 0.0, -12.0]]),
        angle_error_gain_per_s2=np.full((2, 3), 11.5),
        omega_damping_per_s=np.full((2, 3), 8.2),
        omega_bias_rad_s2=np.zeros((2, 3)),
        source_path="synthetic",
        schema="official_visual_dynamics_v1",
    )


def _observation(gate: int, pose=None, rate=(-8.0, 2.0, -3.0)) -> np.ndarray:
    values = np.zeros(32, dtype=np.float64)
    values[6] = 1.0
    values[23] = gate / 6.0
    values[0] = math.tanh(rate[0] / 5.0)
    values[1] = math.tanh(rate[1] / 3.0)
    values[2] = math.tanh(rate[2] / 3.0)
    if pose is not None:
        values[10] = 1.0
        values[11] = math.tanh(pose[0] / 10.0)
        values[12] = math.tanh(pose[1] / 5.0)
        values[13] = math.tanh(pose[2] / 5.0)
        values[14] = math.atan2(pose[1], pose[0]) / (math.pi / 4.0)
    return values


def test_prefix_and_post_gate4_actions_are_exactly_invariant() -> None:
    controller = n232.Gate4MPCController(_ensemble())
    base = [0.123, -0.456, 0.789, -0.234]
    assert controller.apply(_observation(0), base, now_s=1.0) == base
    assert controller.apply(_observation(1), base, now_s=1.1) == base
    assert controller.apply(_observation(2), base, now_s=1.2) == base
    assert controller.apply(_observation(4), base, now_s=1.3) == base
    assert controller.apply(_observation(5), base, now_s=1.4) == base


def test_gate4_replans_on_bounded_fresh_pose_cadence_or_maximum_interval() -> None:
    controller = n232.Gate4MPCController(_ensemble())
    base = [0.0, 0.0, 0.0, 0.0]
    controller.apply(_observation(2, pose=(5.0, 0.0, 0.0)), base, now_s=9.9)
    gate4 = _observation(3, pose=(32.0, 4.0, 4.0))
    first = controller.apply(gate4, base, now_s=10.0)
    assert controller.optimized_plans + controller.failsafe_plans == 1
    second = controller.apply(gate4, base, now_s=10.05)
    assert second == first
    assert controller.cached_action_calls == 1
    early_fresh = _observation(3, pose=(31.9, 3.9, 3.9))
    assert controller.apply(early_fresh, base, now_s=10.10) == first
    assert controller.cached_action_calls == 2
    later_fresh = _observation(3, pose=(31.8, 3.8, 3.8))
    controller.apply(later_fresh, base, now_s=10.19)
    assert controller.optimized_plans + controller.failsafe_plans == 2
    controller.apply(later_fresh, base, now_s=10.45)
    assert controller.optimized_plans + controller.failsafe_plans == 3


def test_gate4_without_coherent_pose_uses_bounded_non_parent_action() -> None:
    controller = n232.Gate4MPCController(_ensemble())
    base = [-1.0, 1.0, -1.0, 1.0]
    action = controller.apply(_observation(3), base, now_s=1.0)
    assert action != base
    assert action[:3] == [0.25, 0.0, 0.0]
    assert all(-1.0 <= value <= 1.0 for value in action)
