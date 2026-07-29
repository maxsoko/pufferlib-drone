import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate2_vertical_margin_n189 as policy


def _observation(
    *, gate=1, visible=True, forward=5.0, closing=8.0, down=-0.2, down_rate=-2.0
):
    values = [0.0] * 32
    values[0] = math.tanh(-closing / 5.0)
    values[2] = math.tanh(down_rate / 3.0)
    values[6] = 1.0
    values[10] = 1.0 if visible else 0.0
    values[11] = math.tanh(forward / 10.0)
    values[13] = math.tanh(down / 5.0)
    values[23] = gate / 6.0
    values[24 + gate] = 1.0
    return values


def test_low_projected_gate2_path_floors_only_thrust():
    controller = policy.Gate2VerticalMarginFloor()
    base = [0.1, -0.2, 0.0, -0.4]
    governed = controller.apply(_observation(down=-0.2, down_rate=-2.0), base)
    assert governed == [0.1, -0.2, 0.3, -0.4]
    assert controller.snapshot()["active"] is True
    assert controller.snapshot()["changed_samples"] == 1


def test_stronger_thrust_is_preserved():
    controller = policy.Gate2VerticalMarginFloor()
    base = [0.1, -0.2, 0.6, -0.4]
    assert controller.apply(_observation(), base) == base
    snapshot = controller.snapshot()
    assert snapshot["active"] is True
    assert snapshot["changed_samples"] == 0


def test_rule_is_stateless_and_releases_on_projection_recovery():
    controller = policy.Gate2VerticalMarginFloor()
    controller.apply(_observation(), [0.0] * 4)
    base = [0.1, -0.2, 0.0, -0.4]
    assert controller.apply(_observation(down=0.5, down_rate=0.0), base) == base
    assert controller.snapshot()["active"] is False


def test_far_or_other_gate_is_exact_noop():
    controller = policy.Gate2VerticalMarginFloor()
    base = [0.1, -0.2, 0.0, -0.4]
    assert controller.apply(_observation(forward=6.1), base) == base
    assert controller.apply(_observation(gate=2), base) == base
    assert controller.apply(_observation(visible=False), base) == base
