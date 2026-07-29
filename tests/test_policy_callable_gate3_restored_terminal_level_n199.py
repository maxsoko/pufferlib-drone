import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate3_restored_terminal_level_n199 as policy


def _observation(
    *, gate=2, visible=True, forward=3.6, closing=8.0,
    right=-0.3, right_rate=0.2, down=0.0, down_rate=0.0,
):
    values = [0.0] * 32
    values[0] = math.tanh(-closing / 5.0)
    values[1] = math.tanh(right_rate / 3.0)
    values[2] = math.tanh(down_rate / 3.0)
    values[6] = 1.0
    values[10] = 1.0 if visible else 0.0
    values[11] = math.tanh(forward / 10.0)
    values[12] = math.tanh(right / 5.0)
    values[13] = math.tanh(down / 5.0)
    values[23] = gate / 6.0
    values[24 + gate] = 1.0
    values[30] = 0.5
    return values


def test_restored_safe_projection_levels_only_roll_and_latches():
    controller = policy.n196_module.Gate3LaterTerminalSafeLevel()
    base = [0.31, -0.8, -0.14, 0.0002]
    assert controller.apply(_observation(), base) == [0.31, 0.0, -0.14, 0.0002]
    assert controller.apply(
        _observation(right=2.0, right_rate=2.0), base
    ) == [0.31, 0.0, -0.14, 0.0002]
    assert controller.snapshot()["activation_count"] == 1


def test_more_than_three_point_seven_metres_remains_inactive():
    controller = policy.n196_module.Gate3LaterTerminalSafeLevel()
    base = [0.31, -0.8, -0.14, 0.0002]
    assert controller.apply(_observation(forward=3.8), base) == base
    assert controller.snapshot()["active"] is False


def test_unsafe_projection_does_not_level_and_transition_releases():
    controller = policy.n196_module.Gate3LaterTerminalSafeLevel()
    base = [0.31, -0.8, -0.14, 0.0002]
    assert controller.apply(_observation(right=1.0), base) == base
    controller.apply(_observation(), base)
    assert controller.apply(_observation(gate=3), base) == base
    assert controller.snapshot()["active"] is False


def test_infer_evaluates_exact_n199_once_before_restored_terminal(monkeypatch):
    observation = _observation()
    events = []
    monkeypatch.setattr(
        policy.n199,
        "infer",
        lambda value: events.append(("n199", value)) or [0.2, -1.0, 0.75, 0.1],
    )
    monkeypatch.setattr(
        policy._RESTORED_TERMINAL,
        "apply",
        lambda value, action: events.append(("terminal", value, action)) or [0.2, 0.0, 0.75, 0.1],
    )
    assert policy.infer(observation) == [0.2, 0.0, 0.75, 0.1]
    assert [event[0] for event in events] == ["n199", "terminal"]
    assert all(event[1] is observation for event in events)
