import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate2_rapid_vertical_deficit_n192 as policy


def _observation(
    *, gate=1, visible=True, forward=5.4, closing=8.0,
    down=0.9, down_rate=-3.0
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


def test_rapid_deficit_with_dormant_thrust_latches_full_thrust_only():
    controller = policy.Gate2RapidVerticalDeficit()
    base = [-0.2, -0.1, 0.0, 0.001]
    governed = controller.apply(_observation(), base)
    assert governed == [-0.2, -0.1, 1.0, 0.001]
    assert controller.snapshot()["activation_count"] == 1
    retained = controller.apply(
        _observation(forward=2.5, down=0.2, down_rate=-3.0),
        [-0.2, -0.1, 0.5, 0.001],
    )
    assert retained == [-0.2, -0.1, 1.0, 0.001]


def test_trigger_rejects_active_base_thrust_or_milder_rate():
    controller = policy.Gate2RapidVerticalDeficit()
    base = [-0.2, -0.1, 0.1, 0.001]
    assert controller.apply(_observation(), base) == base
    assert controller.apply(
        _observation(down_rate=-2.8), [-0.2, -0.1, 0.0, 0.001]
    ) == [-0.2, -0.1, 0.0, 0.001]


def test_projection_recovery_releases_latch():
    controller = policy.Gate2RapidVerticalDeficit()
    controller.apply(_observation(), [-0.2, -0.1, 0.0, 0.001])
    base = [-0.2, -0.1, 0.2, 0.001]
    assert controller.apply(
        _observation(forward=2.0, down=0.2, down_rate=0.0), base
    ) == base
    assert controller.snapshot()["active"] is False


def test_other_gate_or_invisible_releases_latch():
    controller = policy.Gate2RapidVerticalDeficit()
    controller.apply(_observation(), [-0.2, -0.1, 0.0, 0.001])
    base = [-0.2, -0.1, 0.2, 0.001]
    assert controller.apply(_observation(gate=2), base) == base
    controller.apply(_observation(), [-0.2, -0.1, 0.0, 0.001])
    assert controller.apply(_observation(visible=False), base) == base
