import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate3_tight_terminal_level_n201 as policy


def _observation(
    *, gate=2, visible=True, forward=3.1, closing=8.0,
    right=0.2, right_rate=0.0, down=0.0, down_rate=0.0,
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


def test_tight_safe_projection_levels_only_roll_and_latches():
    controller = policy.Gate3TightTerminalSafeLevel()
    base = [0.3, -0.8, -0.1, 0.01]
    assert controller.apply(_observation(right=0.29), base) == [0.3, 0.0, -0.1, 0.01]
    assert controller.apply(_observation(right=1.0), base) == [0.3, 0.0, -0.1, 0.01]
    assert controller.snapshot()["activation_count"] == 1


def test_projection_outside_point_three_metres_remains_correcting():
    controller = policy.Gate3TightTerminalSafeLevel()
    base = [0.3, 0.48, -0.1, 0.01]
    assert controller.apply(_observation(right=-0.3407), base) == base
    assert controller.snapshot()["active"] is False


def test_transition_releases_and_gate1_scope_is_separate():
    controller = policy.Gate3TightTerminalSafeLevel()
    base = [0.3, -0.8, -0.1, 0.01]
    controller.apply(_observation(), base)
    assert controller.apply(_observation(gate=3), base) == base
    assert controller.snapshot()["active"] is False


def test_infer_evaluates_exact_n199_once_then_terminal_then_gate1(monkeypatch):
    observation = _observation()
    events = []
    monkeypatch.setattr(
        policy._N199,
        "infer",
        lambda value: events.append(("n199", value)) or [0.3, -0.8, -0.1, 0.01],
    )
    monkeypatch.setattr(
        policy._TIGHT_TERMINAL,
        "apply",
        lambda value, action: events.append(("terminal", value, action))
        or [0.3, 0.0, -0.1, 0.01],
    )
    monkeypatch.setattr(
        policy._GATE1,
        "apply",
        lambda value, action: events.append(("gate1", value, action)) or action,
    )
    assert policy.infer(observation) == [0.3, 0.0, -0.1, 0.01]
    assert [event[0] for event in events] == ["n199", "terminal", "gate1"]
    assert all(event[1] is observation for event in events)
