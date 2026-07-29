from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate4_identity_locked_intercept_n238 as n238


def _observation(gate: int, pose=None, *, yaw_error_rad: float | None = None):
    values = np.zeros(32, dtype=np.float64)
    values[6] = 1.0
    values[23] = gate / 6.0
    if pose is not None:
        values[10] = 1.0
        values[11] = math.tanh(pose[0] / 10.0)
        values[12] = math.tanh(pose[1] / 5.0)
        values[13] = math.tanh(pose[2] / 5.0)
        error = (
            math.atan2(pose[1], pose[0])
            if yaw_error_rad is None
            else yaw_error_rad
        )
        values[14] = error / (math.pi / 4.0)
    return values


def _acquire(controller, poses=None):
    if poses is None:
        poses = ((30.0, 4.0, 5.0), (29.0, 3.8, 4.8), (28.0, 3.5, 4.5))
    output = None
    for index, pose in enumerate(poses):
        output = controller.apply(
            _observation(3, pose), [0.0] * 4, now_s=1.0 + 0.1 * index
        )
    return output


def test_non_gate4_parent_actions_remain_exact() -> None:
    controller = n238.Gate4IdentityLockedInterceptController()
    parent = [0.123, -0.456, 0.789, -0.234]
    for gate in (0, 1, 2, 4, 5):
        assert controller.apply(
            _observation(gate), parent, now_s=gate / 10.0
        ) == parent


def test_initial_alias_has_no_control_authority_before_coherent_acquisition():
    controller = n238.Gate4IdentityLockedInterceptController()
    output = controller.apply(
        _observation(3, (35.0, 12.0, -8.0), yaw_error_rad=0.7),
        [1.0, -1.0, 1.0, 1.0],
        now_s=1.0,
    )
    assert output[:3] == [n238.BRAKE_PITCH_NORM, 0.0, 0.0]
    assert controller.associated_pose is None


def test_three_coherent_fresh_measurements_acquire_one_family() -> None:
    controller = n238.Gate4IdentityLockedInterceptController()
    output = _acquire(controller)
    assert controller.initial_acquisitions == 1
    assert controller.accepted_fresh == 1
    assert controller.associated_pose is not None
    assert output[0] == n238.BRAKE_PITCH_NORM
    assert output[1] < 0.0
    assert output[2] < 0.0


def test_rejected_alias_cannot_change_any_action_channel() -> None:
    controller = n238.Gate4IdentityLockedInterceptController()
    expected = _acquire(controller)
    aliased = controller.apply(
        _observation(3, (39.0, -15.0, -12.0), yaw_error_rad=-0.7),
        [0.0] * 4,
        now_s=1.25,
    )
    assert aliased == expected
    assert controller.rejected_aliases == 1
    assert controller.associated_pose is not None
    assert controller.associated_pose[0] < 30.0


def test_persistent_farther_family_never_reanchors() -> None:
    controller = n238.Gate4IdentityLockedInterceptController()
    _acquire(controller)
    for index, pose in enumerate(
        ((36.0, -4.0, 4.0), (35.0, -3.8, 4.0), (34.0, -3.6, 4.0))
    ):
        controller.apply(
            _observation(3, pose), [0.0] * 4, now_s=1.4 + 0.1 * index
        )
    assert controller.closer_reacquisitions == 0
    assert controller.associated_pose is not None
    assert controller.associated_pose[0] < 30.0


def test_persistent_closer_family_reacquires_after_three_fresh_samples() -> None:
    controller = n238.Gate4IdentityLockedInterceptController()
    _acquire(controller)
    for index, pose in enumerate(
        ((20.0, 9.0, 5.0), (19.0, 8.7, 4.8), (18.0, 8.4, 4.6))
    ):
        controller.apply(
            _observation(3, pose), [0.0] * 4, now_s=1.4 + 0.1 * index
        )
    assert controller.closer_reacquisitions == 1
    assert controller.associated_pose is not None
    np.testing.assert_allclose(controller.associated_pose, (18.0, 8.4, 4.6))


def test_stale_association_levels_and_brakes_without_raw_alias_authority():
    controller = n238.Gate4IdentityLockedInterceptController()
    _acquire(controller)
    output = controller.apply(
        _observation(3, (38.0, -14.0, -10.0), yaw_error_rad=-0.6),
        [0.0] * 4,
        now_s=1.8,
    )
    assert output[:3] == [n238.BRAKE_PITCH_NORM, 0.0, 0.0]
    assert controller.level_brake_calls > 0


def test_commit_requires_fresh_associated_projected_center_and_is_bounded():
    controller = n238.Gate4IdentityLockedInterceptController()
    output = _acquire(
        controller,
        ((7.8, 0.6, 0.5), (7.5, 0.4, 0.3), (7.2, 0.2, 0.1)),
    )
    assert controller.commit_triggers == 1
    assert output[0] == n238.COMMIT_PITCH_NORM
    dropout = controller.apply(_observation(3), [0.0] * 4, now_s=1.8)
    assert dropout[0] == n238.COMMIT_PITCH_NORM
    expired = controller.apply(_observation(3), [0.0] * 4, now_s=2.3)
    assert expired[:3] == [n238.BRAKE_PITCH_NORM, 0.0, 0.0]
    assert controller.commit_triggers == 1


def test_snapshot_declares_identity_and_no_raw_alias_authority_contract():
    snapshot = n238.Gate4IdentityLockedInterceptController().snapshot()
    contract = snapshot["contract"]
    assert contract["one_target_family_per_gate4_phase"]
    assert contract["rejected_pose_action_authority"] is False
    assert contract["association_loss_action"] == "level_brake"
    assert contract["commit_requires_fresh_associated_projected_center"]
    assert contract["runtime_privileged_state"] is False
    assert contract["gates_1_3_parent_invariant"]
    assert contract["gates_5_6_parent_invariant"]
