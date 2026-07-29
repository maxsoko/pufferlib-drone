import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate3_later_terminal_level_n195 as policy


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
    return values


def test_later_safe_projection_levels_only_roll_and_latches():
    controller = policy.Gate3LaterTerminalSafeLevel()
    base = [0.31, -0.8, -0.14, 0.0002]
    assert controller.apply(_observation(), base) == [0.31, 0.0, -0.14, 0.0002]
    assert controller.apply(
        _observation(right=2.0, right_rate=2.0), base
    ) == [0.31, 0.0, -0.14, 0.0002]
    snapshot = controller.snapshot()
    assert snapshot["activation_count"] == 1
    assert snapshot["changed_samples"] == 2


def test_old_six_metre_admission_is_retired():
    controller = policy.Gate3LaterTerminalSafeLevel()
    base = [0.31, -0.8, -0.14, 0.0002]
    assert controller.apply(_observation(forward=5.0), base) == base
    assert controller.snapshot()["active"] is False


def test_unsafe_projection_does_not_level_and_transition_releases():
    controller = policy.Gate3LaterTerminalSafeLevel()
    base = [0.31, -0.8, -0.14, 0.0002]
    assert controller.apply(_observation(right=1.0), base) == base
    controller.apply(_observation(), base)
    assert controller.apply(_observation(gate=3), base) == base
    assert controller.snapshot()["active"] is False


def test_infer_composes_sanitizer_then_exact_n191_then_three_controllers(monkeypatch):
    events = []
    raw = _observation(gate=1)
    sanitized = list(raw)
    sanitized[10] = 0.0

    monkeypatch.setattr(
        policy._SANITIZER, "sanitize",
        lambda observation: events.append(("sanitize", observation)) or sanitized,
    )
    monkeypatch.setattr(
        policy.n191, "infer",
        lambda observation: events.append(("n191", observation)) or [0.1, 0.2, 0.3, 0.4],
    )

    def apply(name, output):
        def inner(observation, actions):
            events.append((name, observation, list(actions)))
            return output
        return inner

    monkeypatch.setattr(
        policy._LATER_TERMINAL, "apply", apply("terminal", [0.1, 0.0, 0.3, 0.4])
    )
    monkeypatch.setattr(
        policy._RAPID_VERTICAL, "apply", apply("rapid", [0.1, 0.0, 1.0, 0.4])
    )
    monkeypatch.setattr(
        policy._FROZEN_RECOVERY, "apply", apply("frozen", [0.1, 0.0, 1.0, 0.2])
    )
    assert policy.infer(raw) == [0.1, 0.0, 1.0, 0.2]
    assert [event[0] for event in events] == [
        "sanitize", "n191", "terminal", "rapid", "frozen"
    ]
    assert all(event[1] is sanitized for event in events[1:])
