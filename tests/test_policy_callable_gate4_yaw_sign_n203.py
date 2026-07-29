import math

import pytest

import policy_callable_gate4_yaw_sign_n203 as policy


def _observation(*, gate=3, visible=True, yaw=0.0, yaw_error=0.2):
    values = [0.0] * 32
    values[6] = math.cos(yaw / 2.0)
    values[9] = math.sin(yaw / 2.0)
    values[10] = 1.0 if visible else 0.0
    values[14] = yaw_error / (math.pi / 4.0)
    values[23] = gate / 6.0
    values[24 + gate] = 1.0
    return values


def test_positive_camera_error_increases_desired_yaw_only():
    controller = policy.Gate4VisibleYawSign()
    base = [0.2, -0.3, 0.4, -0.7]
    result = controller.apply(
        _observation(yaw=0.1, yaw_error=0.2),
        base,
    )
    assert result == pytest.approx([0.2, -0.3, 0.4, 0.3 / math.pi])


def test_negative_camera_error_decreases_desired_yaw():
    controller = policy.Gate4VisibleYawSign()
    result = controller.apply(
        _observation(yaw=-0.1, yaw_error=-0.2),
        [0.2, -0.3, 0.4, 0.7],
    )
    assert result[3] == pytest.approx(-0.3 / math.pi)


def test_yaw_step_is_bounded_and_absolute_angle_wraps():
    controller = policy.Gate4VisibleYawSign()
    result = controller.apply(
        _observation(yaw=math.pi - 0.1, yaw_error=0.7),
        [0.0, 0.0, 0.0, 0.0],
    )
    assert result[3] == pytest.approx((-math.pi + 0.25) / math.pi)


@pytest.mark.parametrize("kwargs", [{"gate": 2}, {"gate": 4}, {"visible": False}])
def test_non_gate4_or_dropout_is_exact_passthrough(kwargs):
    controller = policy.Gate4VisibleYawSign()
    base = [0.2, -0.3, 0.4, -0.7]
    assert controller.apply(_observation(**kwargs), base) == pytest.approx(base)


def test_snapshot_counts_only_visible_gate4_changes():
    controller = policy.Gate4VisibleYawSign()
    controller.apply(_observation(), [0.0, 0.0, 0.0, 0.0])
    controller.apply(_observation(visible=False), [0.0, 0.0, 0.0, 0.0])
    snapshot = controller.snapshot()
    assert snapshot["active_samples"] == 1
    assert snapshot["changed_samples"] == 1
    assert (
        snapshot["parameters"]["desired_yaw_law"]
        == "current_yaw_plus_camera_yaw_error"
    )
