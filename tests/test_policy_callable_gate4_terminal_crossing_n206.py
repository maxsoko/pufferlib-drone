import math

import numpy as np
import pytest

import policy_callable_gate4_terminal_crossing_n206 as policy


def _observation(
    gate: int = 3,
    *,
    visible: bool = True,
    forward_m: float = 2.0,
    right_m: float = 0.0,
    down_m: float = 0.0,
    yaw_error_rad: float = 0.0,
) -> np.ndarray:
    values = np.zeros(32, dtype=np.float32)
    values[6] = np.float32(1.0)
    values[10] = np.float32(1.0 if visible else 0.0)
    values[11] = np.float32(np.tanh(forward_m / 10.0))
    values[12] = np.float32(np.tanh(right_m / 5.0))
    values[13] = np.float32(np.tanh(down_m / 5.0))
    values[14] = np.float32(yaw_error_rad / (math.pi / 4.0))
    values[23] = np.float32(gate / 6.0)
    if gate < 6:
        values[24 + gate] = np.float32(1.0)
    return values


def test_exact_stage5_near_pass_triggers_one_forward_floor():
    controller = policy.Gate4TerminalCrossingPulse()
    base = [-0.105134, 0.0239125, -0.0387973, 0.0004403]
    result = controller.apply(
        _observation(
            forward_m=1.576642,
            right_m=-0.226642,
            down_m=0.0,
            yaw_error_rad=-0.142772,
        ),
        base,
        now_s=10.0,
    )
    assert result == pytest.approx([0.12, *base[1:]])
    snapshot = controller.snapshot()
    assert snapshot["trigger_count"] == 1
    assert snapshot["changed_samples"] == 1
    assert snapshot["last_trigger_pose"] == pytest.approx(
        {
            "forward_m": 1.576642,
            "right_m": -0.226642,
            "down_m": 0.0,
            "yaw_error_rad": -0.142772,
        },
        abs=1e-5,
    )


@pytest.mark.parametrize(
    "pose",
    [
        {"forward_m": 3.01},
        {"forward_m": 0.0},
        {"right_m": 0.51},
        {"right_m": -0.51},
        {"down_m": 0.51},
        {"down_m": -0.51},
        {"yaw_error_rad": 0.151},
        {"yaw_error_rad": -0.151},
        {"visible": False},
    ],
)
def test_outside_terminal_envelope_is_exact_passthrough(pose):
    controller = policy.Gate4TerminalCrossingPulse()
    base = [-0.2, 0.3, 0.4, -0.5]
    assert controller.apply(_observation(**pose), base, now_s=1.0) == base
    assert controller.snapshot()["trigger_count"] == 0


def test_pulse_bridges_dropout_then_expires_without_retrigger():
    controller = policy.Gate4TerminalCrossingPulse()
    base = [-0.2, 0.3, 0.4, -0.5]
    assert controller.apply(_observation(), base, now_s=1.0)[0] == pytest.approx(0.12)
    dropout = _observation(visible=False)
    assert controller.apply(dropout, base, now_s=1.49)[0] == pytest.approx(0.12)
    assert controller.apply(dropout, base, now_s=1.51) == base
    assert controller.apply(_observation(), base, now_s=2.0) == base
    assert controller.snapshot()["trigger_count"] == 1


def test_already_forward_learned_pitch_is_not_reduced():
    controller = policy.Gate4TerminalCrossingPulse()
    base = [0.4, -0.3, 0.2, -0.1]
    assert controller.apply(_observation(), base, now_s=1.0) == base
    assert controller.snapshot()["changed_samples"] == 0


def test_gate_transition_clears_one_shot_for_next_gate4_phase():
    controller = policy.Gate4TerminalCrossingPulse()
    base = [-0.2, 0.0, 0.0, 0.0]
    controller.apply(_observation(), base, now_s=1.0)
    assert controller.apply(_observation(gate=4), base, now_s=1.1) == base
    assert controller.snapshot()["triggered"] is False
    assert controller.apply(_observation(), base, now_s=2.0)[0] == pytest.approx(0.12)


def test_infer_preserves_n206_outside_changed_pitch(monkeypatch):
    base = [-0.2, 0.3, 0.4, -0.5]
    monkeypatch.setattr(policy.n206, "infer", lambda _values: base)
    policy._CONTROLLER.reset()
    result = policy.infer(_observation())
    assert result == pytest.approx([0.12, 0.3, 0.4, -0.5])


def test_reset_resets_parent_and_pulse(monkeypatch):
    calls = []
    monkeypatch.setattr(policy.n206, "reset", lambda: calls.append("parent"))
    policy._CONTROLLER.apply(_observation(), [0.0] * 4, now_s=1.0)
    policy.reset()
    assert calls == ["parent"]
    assert policy._CONTROLLER.snapshot()["triggered"] is False


def test_snapshot_declares_bounded_coordinate_free_scope():
    parameters = policy.Gate4TerminalCrossingPulse().snapshot()["parameters"]
    assert parameters["one_shot_per_gate4_phase"] is True
    assert parameters["preserve_roll_thrust_yaw"] is True
    assert parameters["course_coordinates_used"] is False
    assert parameters["parent"] == "N206_learned_gate4_tail"
