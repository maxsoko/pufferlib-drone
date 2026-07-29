import numpy as np
import pytest

import policy_callable_gate4_learned_tail_n203 as policy


def _observation(gate: int) -> np.ndarray:
    values = np.zeros(32, dtype=np.float32)
    values[6] = np.float32(1.0)
    values[23] = np.float32(gate / 6.0)
    if gate < 6:
        values[24 + gate] = np.float32(1.0)
    values[30] = np.float32(0.7)
    values[31] = np.float32(0.8)
    return values


class _FakeHysteresis:
    def __init__(self):
        self.cleared = 0

    def _clear_active(self):
        self.cleared += 1


def test_gate4_calls_learned_tail_once_and_clears_parent_phase_state(monkeypatch):
    fake_hysteresis = _FakeHysteresis()
    calls = {"tail": 0, "gate2": 0, "gate3": 0, "gate4": 0}
    captured = {}
    monkeypatch.setattr(policy.n203, "_controller", lambda: fake_hysteresis)
    monkeypatch.setattr(
        policy.n203,
        "infer",
        lambda _values: pytest.fail("N203 inference must be bypassed at Gate 4"),
    )

    def fake_tail(values):
        calls["tail"] += 1
        captured["values"] = values.copy()
        return [1.2, -1.3, 0.4, -0.5]

    monkeypatch.setattr(policy.tail, "infer", fake_tail)
    monkeypatch.setattr(
        policy.hybrid,
        "_clear_gate2_state",
        lambda: calls.__setitem__("gate2", calls["gate2"] + 1),
    )
    monkeypatch.setattr(
        policy.hybrid,
        "_clear_gate3_state",
        lambda: calls.__setitem__("gate3", calls["gate3"] + 1),
    )
    monkeypatch.setattr(
        policy.hybrid,
        "_clear_gate4_state",
        lambda: calls.__setitem__("gate4", calls["gate4"] + 1),
    )

    result = policy.infer(_observation(3))
    assert result == pytest.approx([1.0, -1.0, 0.4, -0.5])
    assert calls == {"tail": 1, "gate2": 1, "gate3": 1, "gate4": 1}
    assert fake_hysteresis.cleared == 1
    assert captured["values"][30:32].tolist() == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize("gate", [0, 1, 2, 4, 5])
def test_every_non_gate4_phase_is_exact_n203_passthrough(monkeypatch, gate):
    expected = [0.2, -0.3, 0.4, -0.5]
    calls = []

    def fake_parent(values):
        calls.append(values.copy())
        return expected

    monkeypatch.setattr(policy.n203, "infer", fake_parent)
    monkeypatch.setattr(
        policy.tail,
        "infer",
        lambda _values: pytest.fail("direct tail is Gate-4-only"),
    )
    assert policy.infer(_observation(gate)) == expected
    assert len(calls) == 1


def test_reset_resets_parent_and_activation_counter(monkeypatch):
    calls = []
    monkeypatch.setattr(policy.n203, "reset", lambda: calls.append("parent"))
    policy._CONTROLLER.record([0.0, 0.0, 0.0, 0.0])
    policy.reset()
    assert calls == ["parent"]
    assert policy._CONTROLLER.snapshot()["active_samples"] == 0


def test_snapshot_declares_learned_coordinate_free_scope():
    snapshot = policy.Gate4LearnedTailDirect().snapshot()
    parameters = snapshot["parameters"]
    assert parameters["manual_gate4_handoff_bypassed"] is True
    assert parameters["gate1_through_gate3_parent"] == "H12_N203"
    assert parameters["course_coordinates_used"] is False
