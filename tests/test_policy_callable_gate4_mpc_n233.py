from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate4_mpc_n233 as n233


class _FakeController:
    def __init__(self) -> None:
        self.times: list[float] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def apply(self, observation, parent_actions, *, now_s=None):
        self.times.append(float(now_s))
        return list(parent_actions)

    def snapshot(self) -> dict:
        return {"times": self.times}


def _observation(gate: int) -> np.ndarray:
    values = np.zeros(32, dtype=np.float64)
    values[6] = 1.0
    values[23] = gate / 6.0
    return values


def test_fixed_clock_preserves_parent_actions_and_validates_no_model_per_tick(
    monkeypatch,
) -> None:
    fake = _FakeController()
    monkeypatch.setattr(n233, "_CONTROLLER", fake)
    monkeypatch.setattr(n233, "_CONTROLLER_MODEL_PATH", Path("synthetic"))
    monkeypatch.setattr(n233, "_MODEL_VALIDATIONS", 1)
    monkeypatch.setattr(n233.n232.parent, "reset", lambda: None)
    monkeypatch.setattr(
        n233.n232.parent,
        "infer",
        lambda _observation: [0.125, -0.25, 0.5, -0.75],
    )

    n233.reset()
    assert n233.infer(_observation(0)) == [0.125, -0.25, 0.5, -0.75]
    assert n233.infer(_observation(1)) == [0.125, -0.25, 0.5, -0.75]
    assert fake.times == [0.0, 1.0 / 60.0]
    assert n233._MODEL_VALIDATIONS == 1


def test_rejects_non_32_value_observation(monkeypatch) -> None:
    monkeypatch.setattr(n233.n232.parent, "infer", lambda _observation: [0.0] * 4)
    try:
        n233.infer([0.0] * 31)
    except ValueError as exc:
        assert "shape (32,)" in str(exc)
    else:
        raise AssertionError("expected shape rejection")
