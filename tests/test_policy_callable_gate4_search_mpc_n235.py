from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gate4_receding_horizon as mpc
import policy_callable_gate4_search_mpc_n235 as n235


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


def _observation(gate: int, pose=None) -> np.ndarray:
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


def test_non_gate4_parent_actions_remain_exact() -> None:
    controller = n235.Gate4SearchMPCController(_ensemble())
    parent = [0.123, -0.456, 0.789, -0.234]
    for gate in (0, 1, 2, 4, 5):
        assert controller.apply(
            _observation(gate), parent, now_s=gate / 10.0
        ) == parent


def test_visible_unadmitted_target_gets_bounded_centering_action() -> None:
    controller = n235.Gate4SearchMPCController(_ensemble())
    action = controller.apply(
        _observation(3, pose=(30.0, 10.0, 8.0)),
        [0.0] * 4,
        now_s=1.0,
    )
    assert action[0] == n235.ACQUISITION_FORWARD_PITCH_NORM
    assert -n235.ACQUISITION_MAX_ROLL_NORM <= action[1] < 0.0
    assert -n235.ACQUISITION_MAX_THRUST_NORM <= action[2] < 0.0
    assert -1.0 <= action[3] < 0.0
    assert controller.visible_acquisition_calls == 1


def test_blind_search_is_time_bounded_then_returns_to_brake() -> None:
    controller = n235.Gate4SearchMPCController(_ensemble())
    controller.apply(
        _observation(3, pose=(30.0, 10.0, 0.0)),
        [0.0] * 4,
        now_s=1.0,
    )
    search = controller.apply(_observation(3), [0.0] * 4, now_s=2.0)
    assert search[0] == n235.ACQUISITION_FORWARD_PITCH_NORM
    assert search[3] < 0.0
    brake = controller.apply(_observation(3), [0.0] * 4, now_s=3.1)
    assert brake[:3] == [n235.n234.NEUTRAL_BRAKE_PITCH_NORM, 0.0, 0.0]
    assert controller.blind_search_calls == 1
    assert controller.blind_brake_calls == 1


def test_visible_near_target_brakes_instead_of_driving_forward() -> None:
    controller = n235.Gate4SearchMPCController(_ensemble())
    action = controller.apply(
        _observation(3, pose=(4.0, 0.2, 0.2)), [0.0] * 4, now_s=1.0
    )
    assert action[0] == n235.n234.NEUTRAL_BRAKE_PITCH_NORM
    assert all(-1.0 <= value <= 1.0 for value in action)
