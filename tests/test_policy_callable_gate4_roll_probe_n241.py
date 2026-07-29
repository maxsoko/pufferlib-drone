from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate4_roll_probe_n241 as n241


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


def _apply(controller, pose, now_s):
    return controller.apply(
        _observation(3, pose), [0.0, 0.0, 0.0, 0.0], now_s=now_s
    )


def _acquire(controller, start_s=1.0):
    output = None
    for index, pose in enumerate(
        ((35.0, 2.0, 3.0), (34.5, 1.9, 2.9), (34.0, 1.8, 2.8))
    ):
        output = _apply(controller, pose, start_s + index * 0.1)
    return output


def test_probe_sequence_is_balanced_bounded_and_completes() -> None:
    controller = n241.Gate4RollProbeController()
    _acquire(controller)
    actions = []
    for index in range(1, 140):
        actions.append(
            _apply(
                controller,
                (33.8 - 0.001 * index, 1.8, 2.8),
                1.2 + index / 60.0,
            )
        )
    excited = [action[1] for action in actions if abs(action[1]) > 0.0]
    assert excited
    assert set(excited) == set(n241.ROLL_SEQUENCE)
    assert max(abs(value) for value in excited) == 0.35
    assert controller.probe_completed
    assert not controller.probe_aborted_near
    assert actions[-1][1] == 0.0


def test_probe_aborts_before_near_region() -> None:
    controller = n241.Gate4RollProbeController()
    first = _acquire(controller)
    assert first[1] == n241.ROLL_SEQUENCE[0]
    near = None
    for index, forward_m in enumerate(np.linspace(33.0, 19.9, 15)):
        near = _apply(controller, (forward_m, 1.0, 2.0), 1.3 + index * 0.1)
    assert near[1] == 0.0
    assert controller.probe_aborted_near


def test_stale_or_rejected_pose_has_zero_probe_authority() -> None:
    controller = n241.Gate4RollProbeController()
    trusted = _acquire(controller)
    assert trusted[1] != 0.0
    alias = _apply(controller, (60.0, 15.0, 10.0), 1.25)
    assert alias[1] == trusted[1]
    stale = controller.apply(
        _observation(3), [1.0, 1.0, 1.0, 1.0], now_s=1.6
    )
    assert stale[1] == 0.0


def test_non_gate4_parent_actions_remain_exact() -> None:
    controller = n241.Gate4RollProbeController()
    parent = [0.123, -0.456, 0.789, -0.234]
    for gate in (0, 1, 2, 4, 5):
        assert controller.apply(
            _observation(gate), parent, now_s=gate / 10.0
        ) == parent
