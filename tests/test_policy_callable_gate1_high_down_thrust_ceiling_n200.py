import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate1_high_down_thrust_ceiling_n200 as policy


def _observation(
    *, gate=0, visible=True, forward=5.0, closing=7.0, down=0.40
):
    values = [0.0] * 32
    values[0] = math.tanh(-closing / 5.0)
    values[6] = 1.0
    values[10] = 1.0 if visible else 0.0
    values[11] = math.tanh(forward / 10.0)
    values[13] = math.tanh(down / 5.0)
    values[23] = gate / 6.0
    values[24 + gate] = 1.0
    values[30] = 0.5
    return values


def test_close_high_down_gate1_caps_only_thrust():
    controller = policy.Gate1HighDownThrustCeiling()
    base = [0.4, 0.2, -0.5, 0.1]
    assert controller.apply(_observation(), base) == [0.4, 0.2, -0.7, 0.1]
    snapshot = controller.snapshot()
    assert snapshot["active_samples"] == 1
    assert snapshot["changed_samples"] == 1


def test_existing_lower_thrust_is_preserved():
    controller = policy.Gate1HighDownThrustCeiling()
    base = [0.4, 0.2, -0.8, 0.1]
    assert controller.apply(_observation(), base) == base
    assert controller.snapshot()["active_samples"] == 1
    assert controller.snapshot()["changed_samples"] == 0


def test_scope_guards_leave_actions_exact():
    controller = policy.Gate1HighDownThrustCeiling()
    base = [0.4, 0.2, -0.5, 0.1]
    cases = (
        _observation(gate=1),
        _observation(visible=False),
        _observation(forward=6.01),
        _observation(closing=0.99),
        _observation(down=0.349),
    )
    for observation in cases:
        assert controller.apply(observation, base) == base
        assert controller.snapshot()["active"] is False


def test_infer_evaluates_exact_n200_once_before_ceiling(monkeypatch):
    observation = _observation()
    events = []
    monkeypatch.setattr(
        policy.n200,
        "infer",
        lambda value: events.append(("n200", value)) or [0.4, 0.2, -0.5, 0.1],
    )
    monkeypatch.setattr(
        policy._CONTROLLER,
        "apply",
        lambda value, action: events.append(("ceiling", value, action))
        or [0.4, 0.2, -0.7, 0.1],
    )
    assert policy.infer(observation) == [0.4, 0.2, -0.7, 0.1]
    assert [event[0] for event in events] == ["n200", "ceiling"]
    assert all(event[1] is observation for event in events)
