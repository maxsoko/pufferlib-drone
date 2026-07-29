from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gate4_receding_horizon as mpc


def _ensemble() -> mpc.VisualDynamicsEnsemble:
    members = 3
    return mpc.VisualDynamicsEnsemble(
        thrust_gain=np.asarray(
            [[60.0, 15.0, 40.0], [65.0, 17.0, 43.0], [70.0, 19.0, 45.0]]
        ),
        linear_drag_per_s=np.asarray(
            [[0.0, 0.0, 0.75], [0.0, 0.1, 0.80], [0.0, 0.2, 0.85]]
        ),
        acceleration_bias_m_s2=np.asarray(
            [[0.0, 0.0, -11.0], [0.0, 0.0, -11.8], [0.0, 0.0, -12.3]]
        ),
        angle_error_gain_per_s2=np.full((members, 3), 11.5),
        omega_damping_per_s=np.full((members, 3), 8.2),
        omega_bias_rad_s2=np.zeros((members, 3)),
        source_path="synthetic",
        schema="official_visual_dynamics_v1",
    )


def _observation(
    *,
    gate: int,
    pose: tuple[float, float, float] | None = None,
    relative_rate: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    values = np.zeros(32, dtype=np.float64)
    values[6] = 1.0
    values[23] = gate / 6.0
    values[0] = math.tanh(relative_rate[0] / 5.0)
    values[1] = math.tanh(relative_rate[1] / 3.0)
    values[2] = math.tanh(relative_rate[2] / 3.0)
    if pose is not None:
        values[10] = 1.0
        values[11] = math.tanh(pose[0] / 10.0)
        values[12] = math.tanh(pose[1] / 5.0)
        values[13] = math.tanh(pose[2] / 5.0)
        values[14] = math.atan2(pose[1], pose[0]) / (math.pi / 4.0)
    return values


def test_tracker_carries_observable_gate3_velocity_into_gate4() -> None:
    tracker = mpc.Gate4VisualStateTracker()
    # Vehicle world velocity [8,-2,-3] means the fixed Gate-3 vector changes
    # at [-8,+2,+3] in Z-up, or down-rate -3 in the NED observation.
    gate3 = _observation(gate=2, pose=(5.0, 0.0, 0.0), relative_rate=(-8.0, 2.0, -3.0))
    tracker.observe_prefix(gate3, gate_index=2)
    state, fresh = tracker.update_gate4(
        _observation(gate=3, pose=(32.0, 4.0, 4.0)), now_s=10.0
    )
    assert fresh
    assert state is not None
    np.testing.assert_allclose(state.velocity_world_m_s, [8.0, -2.0, -3.0], atol=1e-6)
    np.testing.assert_allclose(state.reference_ned_m, [32.0, 4.0, 4.0], atol=1e-5)


def test_tracker_rejects_far_initial_alias() -> None:
    tracker = mpc.Gate4VisualStateTracker()
    state, fresh = tracker.update_gate4(
        _observation(gate=3, pose=(70.0, 0.0, 0.0)), now_s=1.0
    )
    assert state is None
    assert not fresh
    assert tracker.rejected_samples == 1


def test_optimizer_is_deterministic_and_stale_state_fails_closed() -> None:
    ensemble = _ensemble()
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    state = mpc.VisualState(
        relative_world_m=np.asarray([32.0, 4.0, -4.0]),
        velocity_world_m_s=np.asarray([8.0, -2.0, -3.0]),
        euler_rpy_rad=np.zeros(3),
        omega_rpy_rad_s=np.zeros(3),
        body_ned_m=np.asarray([32.0, 4.0, 4.0]),
        reference_quaternion_wxyz=quaternion,
        reference_ned_m=np.asarray([32.0, 4.0, 4.0]),
        position_uncertainty_m=0.2,
        measurement_age_s=0.0,
    )
    config = mpc.OptimizerConfig(candidates=24, iterations=1)
    left = mpc.Gate4RecedingHorizonOptimizer(ensemble, config).plan(
        state, yaw_target_rad=0.0
    )
    right = mpc.Gate4RecedingHorizonOptimizer(ensemble, config).plan(
        state, yaw_target_rad=0.0
    )
    assert left == right
    assert all(-1.0 <= value <= 1.0 for value in left.action)

    stale = state.copy()
    stale.measurement_age_s = 0.31
    rejected = mpc.Gate4RecedingHorizonOptimizer(ensemble, config).plan(
        stale, yaw_target_rad=0.0
    )
    assert not rejected.accepted
    assert rejected.reason == "measurement_stale"


def test_propagation_terminates_in_finite_observable_state() -> None:
    ensemble = _ensemble()
    state = mpc.VisualState(
        relative_world_m=np.asarray([10.0, 1.0, -1.0]),
        velocity_world_m_s=np.asarray([5.0, 0.0, -1.0]),
        euler_rpy_rad=np.zeros(3),
        omega_rpy_rad_s=np.zeros(3),
        body_ned_m=np.asarray([10.0, 1.0, 1.0]),
        reference_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        reference_ned_m=np.asarray([10.0, 1.0, 1.0]),
        position_uncertainty_m=0.1,
        measurement_age_s=0.0,
    )
    following = mpc.propagate_member(
        state, (0.2, -0.4, 0.1, 0.0), ensemble, member=1, duration_s=0.1
    )
    assert np.all(np.isfinite(following.relative_world_m))
    assert following.reference_ned_m[0] < state.reference_ned_m[0]

