import math

import numpy as np
import pytest

import policy_callable_gate4_coherent_target_hold_n203 as policy


def _encode(value: float, scale: float) -> np.float32:
    return np.float32(np.tanh(value / scale))


def _observation(
    *,
    gate: int = 3,
    visible: bool = True,
    yaw: float = 0.0,
    forward_m: float = 20.0,
    right_m: float = 2.0,
    down_m: float = 3.0,
) -> np.ndarray:
    values = np.zeros(32, dtype=np.float32)
    values[6] = np.float32(math.cos(yaw / 2.0))
    values[9] = np.float32(math.sin(yaw / 2.0))
    values[10] = np.float32(1.0 if visible else 0.0)
    values[11] = _encode(forward_m, 10.0)
    values[12] = _encode(right_m, 5.0)
    values[13] = _encode(down_m, 5.0)
    values[23] = np.float32(gate / 6.0)
    values[24 + gate] = np.float32(1.0)
    return values


def test_visible_rejected_pose_holds_previous_coherent_yaw_only():
    controller = policy.Gate4CoherentTargetHold()
    result = controller.apply(
        _observation(yaw=0.1),
        [0.2, -0.3, 0.4, -0.7],
        had_anchor=True,
        previous_coherent_yaw_target_rad=0.3,
        current_pose_associated=False,
    )
    assert result == pytest.approx([0.2, -0.3, 0.4, 0.3 / math.pi])


def test_held_delta_is_bounded_and_absolute_angle_wraps():
    controller = policy.Gate4CoherentTargetHold()
    result = controller.apply(
        _observation(yaw=math.pi - 0.1),
        [0.0, 0.0, 0.0, 0.0],
        had_anchor=True,
        previous_coherent_yaw_target_rad=-2.0,
        current_pose_associated=False,
    )
    assert result[3] == pytest.approx((-math.pi + 0.25) / math.pi)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"had_anchor": False, "current_pose_associated": False},
        {"had_anchor": True, "current_pose_associated": True},
    ],
)
def test_pre_anchor_and_associated_visible_samples_are_passthrough(kwargs):
    controller = policy.Gate4CoherentTargetHold()
    base = [0.2, -0.3, 0.4, -0.7]
    result = controller.apply(
        _observation(),
        base,
        previous_coherent_yaw_target_rad=0.3,
        **kwargs,
    )
    assert result == pytest.approx(base)


@pytest.mark.parametrize("observation", [_observation(gate=2), _observation(visible=False)])
def test_non_gate4_and_dropout_are_passthrough(observation):
    controller = policy.Gate4CoherentTargetHold()
    base = [0.2, -0.3, 0.4, -0.7]
    result = controller.apply(
        observation,
        base,
        had_anchor=True,
        previous_coherent_yaw_target_rad=0.3,
        current_pose_associated=False,
    )
    assert result == pytest.approx(base)


def test_infer_restores_parent_target_after_rejected_visible_pose(monkeypatch):
    observation = _observation(yaw=0.1)
    monkeypatch.setattr(policy.hybrid, "_GATE4_ANCHOR_VECTOR", (30.0, 2.0, 3.0))
    monkeypatch.setattr(policy.hybrid, "_GATE4_LAST_VISIBLE_YAW_TARGET_RAD", 0.3)
    monkeypatch.setattr(policy.hybrid, "_GATE4_LAST_VISIBLE_YAW_TARGET_S", 1.0)

    def fake_parent(_observation):
        policy.hybrid._GATE4_LAST_VECTOR = (8.0, -4.0, 1.0)
        policy.hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_RAD = -1.2
        return [0.2, -0.3, 0.4, -0.7]

    monkeypatch.setattr(policy.n203, "infer", fake_parent)
    result = policy.infer(observation)
    assert result == pytest.approx([0.2, -0.3, 0.4, 0.3 / math.pi])
    assert policy.hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_RAD == pytest.approx(0.3)
    assert policy.hybrid._GATE4_LAST_VISIBLE_YAW_TARGET_S > 1.0


def test_snapshot_reports_bounded_scope():
    controller = policy.Gate4CoherentTargetHold()
    controller.apply(
        _observation(),
        [0.0, 0.0, 0.0, 0.0],
        had_anchor=True,
        previous_coherent_yaw_target_rad=0.2,
        current_pose_associated=False,
    )
    snapshot = controller.snapshot()
    assert snapshot["held_samples"] == 1
    assert snapshot["parameters"]["visible_rejected_only"] is True
    assert snapshot["parameters"]["preserve_pitch_roll_thrust"] is True
