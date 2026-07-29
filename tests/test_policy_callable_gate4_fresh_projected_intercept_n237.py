from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gate4_receding_horizon as mpc
import policy_callable_gate4_fresh_projected_intercept_n237 as n237


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


def _prime(controller, monkeypatch, plan: mpc.PlanResult):
    monkeypatch.setattr(controller.optimizer, "plan", lambda *_args, **_kwargs: plan)
    output = None
    parent = [0.0, 0.0, 0.0, 0.0]
    for index, pose in enumerate(
        ((30.0, 5.0, 6.0), (29.0, 4.8, 5.8), (28.0, 4.5, 5.5))
    ):
        output = controller.apply(
            _observation(3, pose=pose), parent, now_s=1.0 + 0.1 * index
        )
    assert output is not None
    return output


def test_non_gate4_parent_actions_remain_exact() -> None:
    controller = n237.Gate4FreshProjectedInterceptController(_ensemble())
    parent = [0.123, -0.456, 0.789, -0.234]
    for gate in (0, 1, 2, 4, 5):
        assert controller.apply(
            _observation(gate), parent, now_s=gate / 10.0
        ) == parent


def test_fresh_state_or_prediction_rejection_uses_projected_intercept(
    monkeypatch,
) -> None:
    for reason in sorted(n237.PROJECTED_INTERCEPT_REASONS):
        controller = n237.Gate4FreshProjectedInterceptController(_ensemble())
        projected = mpc.PlanResult(
            action=(0.8, -0.9, -0.7, -0.05),
            accepted=False,
            reason=reason,
            horizon_s=1.0,
            objective=2.0,
            ensemble_disagreement_m=1.0,
            predicted_terminal_body_ned_m=None,
            crossing_members=0,
        )
        output = _prime(controller, monkeypatch, projected)
        assert output == list(projected.action)
        assert controller.fresh_projected_intercept_plans == 1
        assert controller.rejected_plan_neutralizations == 0


def test_unapproved_rejection_keeps_n236_staged_brake(monkeypatch) -> None:
    controller = n237.Gate4FreshProjectedInterceptController(_ensemble())
    rejected = mpc.PlanResult(
        action=(1.0, -0.9, -1.0, 0.5),
        accepted=False,
        reason="measurement_stale",
        horizon_s=0.0,
        objective=None,
        ensemble_disagreement_m=None,
        predicted_terminal_body_ned_m=None,
        crossing_members=0,
    )
    output = _prime(controller, monkeypatch, rejected)
    assert output[0] == n237.n236.n235.n234.NEUTRAL_BRAKE_PITCH_NORM
    assert output[1] < 0.0
    assert output[2] < 0.0
    assert controller.fresh_projected_intercept_plans == 0
    assert controller.rejected_plan_neutralizations == 1


def test_stale_projected_action_expires_to_n236_brake(monkeypatch) -> None:
    controller = n237.Gate4FreshProjectedInterceptController(_ensemble())
    projected = mpc.PlanResult(
        action=(1.0, -0.9, -1.0, 0.5),
        accepted=False,
        reason="state_uncertain",
        horizon_s=0.0,
        objective=None,
        ensemble_disagreement_m=1.0,
        predicted_terminal_body_ned_m=None,
        crossing_members=0,
    )
    _prime(controller, monkeypatch, projected)
    output = controller.apply(_observation(3), [0.0] * 4, now_s=1.6)
    assert output[:3] == [
        n237.n236.n235.n234.NEUTRAL_BRAKE_PITCH_NORM,
        0.0,
        0.0,
    ]
    assert controller.track_resets == 1


def test_range_aliases_cannot_overwrite_action_propagated_velocity(
    monkeypatch,
) -> None:
    controller = n237.Gate4FreshProjectedInterceptController(_ensemble())
    controller.apply(
        _observation(2, pose=(4.0, 0.0, 0.0), rate=(-8.0, 2.0, -3.0)),
        [0.0] * 4,
        now_s=0.9,
    )
    accepted = mpc.PlanResult(
        action=(0.5, -0.9, 0.4, 0.0),
        accepted=True,
        reason="optimized",
        horizon_s=1.0,
        objective=1.0,
        ensemble_disagreement_m=0.1,
        predicted_terminal_body_ned_m=(0.0, 0.0, 0.0),
        crossing_members=2,
    )
    monkeypatch.setattr(
        controller.optimizer, "plan", lambda *_args, **_kwargs: accepted
    )
    for index, pose in enumerate(
        ((68.0, -6.0, 5.0), (32.0, 3.0, 5.0), (33.0, 3.5, 5.5), (34.0, 4.0, 6.0))
    ):
        controller.apply(
            _observation(3, pose=pose), [0.0] * 4, now_s=1.0 + index * 0.1
        )
    velocity = controller.observer_velocity_world_m_s
    assert velocity is not None
    assert 0.0 < velocity[0] < 20.0
    assert abs(velocity[1]) < 5.0
    assert abs(velocity[2]) < 10.0
    np.testing.assert_allclose(controller.tracker.velocity_world_m_s, velocity)
    assert controller.tracker.rejected_samples >= 1


def test_snapshot_declares_fresh_observable_contract() -> None:
    controller = n237.Gate4FreshProjectedInterceptController(_ensemble())
    contract = controller.snapshot()["fresh_projected_intercept"]["contract"]
    assert contract["requires_fresh_measurement"]
    assert contract["requires_coherent_track"]
    assert contract["runtime_privileged_state"] is False
    observer = controller.snapshot()["action_propagated_observer"]["contract"]
    assert observer["gate3_entry_velocity_prior"]
    assert observer["action_history_propagation"]
    assert observer["camera_corrects_position_not_velocity"]
    assert observer["runtime_privileged_state"] is False
    assert controller.optimizer.config.uncertainty_limit_m == 2.0
