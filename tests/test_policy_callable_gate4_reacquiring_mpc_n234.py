from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gate4_receding_horizon as mpc
import policy_callable_gate4_reacquiring_mpc_n234 as n234


def _ensemble() -> mpc.VisualDynamicsEnsemble:
    return mpc.VisualDynamicsEnsemble(
        thrust_gain=np.asarray([[60.0, 15.0, 40.0], [70.0, 18.0, 45.0]]),
        linear_drag_per_s=np.asarray([[0.0, 0.0, 0.75], [0.0, 0.2, 0.85]]),
        acceleration_bias_m_s2=np.asarray(
            [[0.0, 0.0, -11.0], [0.0, 0.0, -12.0]]
        ),
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


def test_non_gate4_parent_actions_remain_exact() -> None:
    controller = n234.Gate4ReacquiringMPCController(_ensemble())
    parent = [0.123, -0.456, 0.789, -0.234]
    for gate in (0, 1, 2, 4, 5):
        assert controller.apply(
            _observation(gate), parent, now_s=gate / 10.0
        ) == parent


def test_requires_coherent_fresh_samples_and_accepts_far_gate() -> None:
    controller = n234.Gate4ReacquiringMPCController(_ensemble())
    parent = [-1.0, 1.0, -1.0, 1.0]
    first = controller.apply(
        _observation(3, pose=(68.0, 1.0, 1.0)), parent, now_s=1.0
    )
    assert first[:3] == [0.25, 0.0, 0.0]
    assert controller.tracker.accepted_samples == 1
    controller.apply(
        _observation(3, pose=(67.5, 1.0, 1.0)), parent, now_s=1.1
    )
    assert controller.coherent_fresh_samples == 2
    controller.apply(
        _observation(3, pose=(67.0, 1.0, 1.0)), parent, now_s=1.2
    )
    assert controller.coherent_fresh_samples == 3
    assert controller.optimized_plans + controller.rejected_plans == 1


def test_stale_aggressive_action_is_expired_and_track_reacquires() -> None:
    controller = n234.Gate4ReacquiringMPCController(_ensemble())
    parent = [0.0, 0.0, 0.0, 0.0]
    for index, pose in enumerate(
        ((30.0, 2.0, 2.0), (29.5, 2.0, 2.0), (29.0, 2.0, 2.0))
    ):
        controller.apply(_observation(3, pose=pose), parent, now_s=1.0 + 0.1 * index)
    controller.cached_action = (1.0, -0.9, 1.0, 0.5)
    action = controller.apply(_observation(3), parent, now_s=1.6)
    assert action[:3] == [0.25, 0.0, 0.0]
    assert controller.track_resets == 1
    assert controller.cached_action == (0.25, 0.0, 0.0, 0.0)


def test_rejected_optimizer_plan_uses_neutral_action(monkeypatch) -> None:
    controller = n234.Gate4ReacquiringMPCController(_ensemble())
    rejected = mpc.PlanResult(
        action=(1.0, -0.9, 1.0, 0.5),
        accepted=False,
        reason="state_uncertain",
        horizon_s=0.0,
        objective=None,
        ensemble_disagreement_m=2.0,
        predicted_terminal_body_ned_m=None,
        crossing_members=0,
    )
    monkeypatch.setattr(controller.optimizer, "plan", lambda *_args, **_kwargs: rejected)
    parent = [0.0, 0.0, 0.0, 0.0]
    output = None
    for index, pose in enumerate(
        ((30.0, 1.0, 1.0), (29.5, 1.0, 1.0), (29.0, 1.0, 1.0))
    ):
        output = controller.apply(
            _observation(3, pose=pose), parent, now_s=1.0 + 0.1 * index
        )
    assert output is not None
    assert output[:3] == [0.25, 0.0, 0.0]
    assert controller.plan_rejection_counts == {"state_uncertain": 1}


def test_wrapper_uses_fixed_clock_and_preserves_parent(monkeypatch) -> None:
    class FakeController:
        def __init__(self) -> None:
            self.times = []

        def reset(self) -> None:
            self.times.clear()

        def apply(self, _observation, parent, *, now_s):
            self.times.append(float(now_s))
            return list(parent)

        def snapshot(self):
            return {"times": self.times}

    fake = FakeController()
    monkeypatch.setattr(n234, "_CONTROLLER", fake)
    monkeypatch.setattr(n234.n233.n232.parent, "reset", lambda: None)
    monkeypatch.setattr(
        n234.n233.n232.parent,
        "infer",
        lambda _observation: [0.125, -0.25, 0.5, -0.75],
    )
    n234.reset()
    assert n234.infer(_observation(0)) == [0.125, -0.25, 0.5, -0.75]
    assert n234.infer(_observation(1)) == [0.125, -0.25, 0.5, -0.75]
    assert fake.times == [0.0, 1.0 / 60.0]
