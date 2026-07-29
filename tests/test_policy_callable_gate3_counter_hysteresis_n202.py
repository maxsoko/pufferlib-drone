import importlib

import numpy as np
import pytest

import policy_callable_six_gate_composite as tail


def _encode(value: float, scale: float) -> np.float32:
    return np.float32(np.tanh(value / scale))


def _observation(
    *,
    gate: int = 2,
    visible: bool = True,
    forward_m: float = 8.0,
    right_m: float = -1.0,
    closing_m_s: float = 6.0,
    right_rate_m_s: float = 1.0,
) -> np.ndarray:
    values = np.zeros(32, dtype=np.float32)
    values[23] = np.float32(gate / 6.0)
    values[24 + gate] = np.float32(1.0)
    values[10] = np.float32(1.0 if visible else 0.0)
    values[0] = _encode(-closing_m_s, 5.0)
    values[1] = _encode(right_rate_m_s, 3.0)
    values[11] = _encode(forward_m, 10.0)
    values[12] = _encode(right_m, 5.0)
    return values


def _module(monkeypatch: pytest.MonkeyPatch, threshold: str = "-1.2"):
    monkeypatch.setenv("PUFFER_GATE3_COUNTER_ENTRY_M", threshold)
    import policy_callable_gate3_counter_hysteresis_n202 as module

    module._CONTROLLER = None
    return importlib.reload(module)


def test_requires_one_of_three_frozen_entry_thresholds(monkeypatch):
    monkeypatch.delenv("PUFFER_GATE3_COUNTER_ENTRY_M", raising=False)
    import policy_callable_gate3_counter_hysteresis_n202 as module

    module._CONTROLLER = None
    with pytest.raises(RuntimeError, match="is required"):
        module._controller()
    monkeypatch.setenv("PUFFER_GATE3_COUNTER_ENTRY_M", "-1.0")
    with pytest.raises(RuntimeError, match="must be one of"):
        module._controller()


def test_latches_counter_and_releases_with_fixed_gap(monkeypatch):
    module = _module(monkeypatch, "-1.2")
    controller = module.Gate3CounterHysteresis()
    base = [0.2, 0.7, 0.3, -0.1]

    # projection = -1 + 1 * (8 - 2) / 6 = 0, entering counter.
    entered = controller.apply(_observation(), base)
    assert entered == pytest.approx([0.2, -1.0, 0.3, -0.1])
    assert controller.active

    # Projection -1.5 lies below entry but above release (-2.0), so hold.
    held = controller.apply(
        _observation(right_m=-2.5, right_rate_m_s=1.0), base
    )
    assert held[1] == -1.0
    assert controller.active

    # Projection -2.2 crosses the release threshold and restores N202 output.
    released = controller.apply(
        _observation(right_m=-3.2, right_rate_m_s=1.0), base
    )
    assert released == pytest.approx(base)
    assert not controller.active
    assert controller.activation_count == 1
    assert controller.release_count == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gate": 1},
        {"visible": False},
        {"forward_m": 4.0},
        {"forward_m": 15.1},
        {"closing_m_s": 0.9},
    ],
)
def test_is_exact_passthrough_outside_early_gate3_window(monkeypatch, kwargs):
    module = _module(monkeypatch)
    controller = module.Gate3CounterHysteresis()
    base = [0.2, 0.7, 0.3, -0.1]
    assert controller.apply(_observation(**kwargs), base) == pytest.approx(base)
    assert not controller.active


def test_all_outputs_are_clamped_and_nonroll_actions_are_preserved(monkeypatch):
    module = _module(monkeypatch, "-1.6")
    controller = module.Gate3CounterHysteresis()
    actions = controller.apply(_observation(), [2.0, 2.0, -2.0, -3.0])
    assert actions == pytest.approx([1.0, -1.0, -1.0, -1.0])
    assert all(-1.0 <= value <= 1.0 for value in actions)
    assert tail._gate_index(_observation()) == 2
