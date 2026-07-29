import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate3_terminal_level_n191 as policy


def _observation(
    *, gate=2, visible=True, forward=3.6, closing=8.0,
    right=-0.3, right_rate=0.2, down=0.0, down_rate=0.0
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


def test_safe_projection_levels_only_roll_and_latches():
    controller = policy.Gate3TerminalSafeLevel()
    base = [0.31, -0.8, -0.14, 0.0002]
    assert controller.apply(_observation(), base) == [0.31, 0.0, -0.14, 0.0002]
    retained = controller.apply(
        _observation(right=2.0, right_rate=2.0), base
    )
    assert retained == [0.31, 0.0, -0.14, 0.0002]
    assert controller.snapshot()["activation_count"] == 1
    assert controller.snapshot()["changed_samples"] == 2


def test_unsafe_projection_does_not_level():
    controller = policy.Gate3TerminalSafeLevel()
    base = [0.31, -0.8, -0.14, 0.0002]
    assert controller.apply(_observation(right=1.0), base) == base
    assert controller.snapshot()["active"] is False


def test_rule_releases_on_official_gate_transition():
    controller = policy.Gate3TerminalSafeLevel()
    base = [0.31, -0.8, -0.14, 0.0002]
    controller.apply(_observation(), base)
    assert controller.apply(_observation(gate=3), base) == base
    assert controller.snapshot()["active"] is False


def test_far_invisible_and_invalid_contracts():
    controller = policy.Gate3TerminalSafeLevel()
    base = [0.31, -0.8, -0.14, 0.0002]
    assert controller.apply(_observation(forward=6.1), base) == base
    assert controller.apply(_observation(visible=False), base) == base
    try:
        controller.apply([0.0] * 31, base)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid observation shape accepted")
