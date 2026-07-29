import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_n225_gate3_vertical_guard as policy


def _observation(*, gate=2, forward=10.0, closing=8.0, down=0.0, down_rate=-2.0):
    values = [0.0] * 32
    values[0] = math.tanh(-closing * 0.2)
    values[2] = math.tanh(down_rate / 3.0)
    values[6] = 1.0
    values[10] = 1.0
    values[11] = math.tanh(forward * 0.1)
    values[13] = math.tanh(down * 0.2)
    values[23] = gate / 6.0
    values[24 + gate] = 1.0
    return values


def test_guard_floors_only_strong_subhover_projected_vertical_miss():
    controller = policy.Gate3VerticalThrustGuard()
    base = [0.31, -0.42, -0.36, 0.07]

    governed = controller.apply(_observation(), base)

    assert governed == pytest.approx([0.31, -0.42, 0.0, 0.07])
    snapshot = controller.snapshot()
    assert snapshot["activation_count"] == 1
    assert snapshot["changed_samples"] == 1
    assert snapshot["last_projected_down_m"] < 0.0


def test_guard_does_not_change_moderate_thrust_or_safe_projection():
    controller = policy.Gate3VerticalThrustGuard()
    assert controller.apply(_observation(), [0.1, 0.2, -0.1, 0.3]) == pytest.approx(
        [0.1, 0.2, -0.1, 0.3]
    )
    assert controller.apply(
        _observation(down=3.0, down_rate=0.0), [0.1, 0.2, -0.4, 0.3]
    ) == pytest.approx([0.1, 0.2, -0.4, 0.3])


def test_guard_latches_for_gate3_and_resets_on_official_transition():
    controller = policy.Gate3VerticalThrustGuard()
    controller.apply(_observation(), [0.0, 0.0, -0.4, 0.0])
    assert controller.apply(
        _observation(down=3.0, down_rate=0.0), [0.2, 0.3, -0.1, 0.4]
    ) == pytest.approx([0.2, 0.3, 0.0, 0.4])
    assert controller.apply(
        _observation(gate=3), [0.2, 0.3, -0.1, 0.4]
    ) == pytest.approx([0.2, 0.3, -0.1, 0.4])
    assert controller.snapshot()["active"] is False
