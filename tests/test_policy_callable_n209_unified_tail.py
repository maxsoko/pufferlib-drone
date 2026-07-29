from pathlib import Path

import numpy as np
import pytest

import policy_callable_n209_unified_tail as policy


def _observation(gate: int) -> np.ndarray:
    values = np.linspace(-0.9, 0.9, 32, dtype=np.float32)
    values[6:10] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    values[23] = np.float32(gate / 6.0)
    values[24:30] = np.float32(0.0)
    if gate < 6:
        values[24 + gate] = np.float32(1.0)
    values[30] = np.float32(0.73)
    values[31] = np.float32(0.81)
    return values


class _FakeTail:
    def __init__(self) -> None:
        self.reset_count = 0
        self.observations = []

    def reset_state(self) -> None:
        self.reset_count += 1

    def infer(self, observation):
        self.observations.append(np.asarray(observation).copy())
        return [0.1, -0.2, 0.3, -0.4]


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setenv("PUFFER_GATE3_COUNTER_ENTRY_M", "-1.2")
    monkeypatch.setattr(policy, "_TAIL", None)
    monkeypatch.setattr(policy, "_TAIL_PATH", None)
    monkeypatch.setattr(policy, "_TAIL_ACTIVE", False)


@pytest.mark.parametrize("gate", [0, 1, 2])
def test_gates_one_through_three_are_exact_h12_passthrough(monkeypatch, gate):
    expected = [0.2, -0.3, 0.4, -0.5]
    captured = []
    monkeypatch.setattr(
        policy.h12, "infer", lambda values: captured.append(values.copy()) or expected
    )
    monkeypatch.setattr(
        policy,
        "_resolve_tail",
        lambda: pytest.fail("N209 must not load during the prefix"),
    )

    observation = _observation(gate)
    assert policy.infer(observation) == expected
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], observation)


def test_gate4_resets_once_then_one_recurrent_tail_owns_gates4_through6(monkeypatch):
    tail = _FakeTail()
    monkeypatch.setattr(policy, "_resolve_tail", lambda: tail)
    monkeypatch.setattr(
        policy.h12,
        "infer",
        lambda _values: pytest.fail("H12 must stop after official Gate 3"),
    )

    for gate in (3, 3, 4, 5):
        assert policy.infer(_observation(gate)) == pytest.approx(
            [0.1, -0.2, 0.3, -0.4]
        )

    assert tail.reset_count == 1
    assert len(tail.observations) == 4
    for values in tail.observations:
        assert values[24:32].tolist() == pytest.approx([0.0] * 8)
    assert tail.observations[0][23] == pytest.approx(3.0 / 6.0)
    assert tail.observations[2][23] == pytest.approx(4.0 / 6.0)
    assert tail.observations[3][23] == pytest.approx(5.0 / 6.0)


def test_return_to_prefix_clears_tail_state(monkeypatch):
    tail = _FakeTail()
    monkeypatch.setattr(policy, "_TAIL", tail)
    monkeypatch.setattr(policy, "_resolve_tail", lambda: tail)
    monkeypatch.setattr(policy.h12, "infer", lambda _values: [0.0] * 4)

    policy.infer(_observation(3))
    policy.infer(_observation(1))
    assert tail.reset_count == 2
    assert policy._TAIL_ACTIVE is False


def test_reset_resets_prefix_and_loaded_tail(monkeypatch):
    calls = []
    tail = _FakeTail()
    monkeypatch.setattr(policy, "_TAIL", tail)
    monkeypatch.setattr(policy, "_TAIL_ACTIVE", True)
    monkeypatch.setattr(policy.h12, "reset", lambda: calls.append("h12"))

    policy.reset()
    assert calls == ["h12"]
    assert tail.reset_count == 1
    assert policy._TAIL_ACTIVE is False


def test_checkpoint_and_parent_are_immutable_hash_pins():
    assert policy.EXPECTED_N209_CHECKPOINT_SHA256 == (
        "efad0967b46d10bd30eebf03c85a6bc2cecdfb9646bab9852a0d5c701357aefb"
    )
    assert policy.EXPECTED_H12_SOURCE_SHA256 == (
        "b4ccbcb90f30ce866c8a9f827c98d457708ca925345cd1847a6d3ef3ebf80f63"
    )
    assert Path(policy.h12.__file__).is_file()


def test_snapshot_declares_one_unwrapped_late_tail():
    parameters = policy.controller_snapshot()["n210_unified_tail"]["parameters"]
    assert parameters["prefix_gate_indices"] == [0, 1, 2]
    assert parameters["n209_tail_gate_indices"] == [3, 4, 5]
    assert parameters["n209_zero_state_at_gate4"] is True
    assert parameters["manual_late_gate_action_wrappers"] is False
    assert parameters["course_coordinates_used"] is False
