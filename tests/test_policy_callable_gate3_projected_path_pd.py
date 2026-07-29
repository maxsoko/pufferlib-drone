import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate3_projected_path_pd as policy


def _observation(
    *,
    gate: int = 2,
    forward: float = 10.0,
    right: float = -2.0,
    closing: float = 8.0,
    right_rate: float = 0.0,
    visible: bool = True,
) -> list[float]:
    values = [0.0] * 32
    values[0] = math.tanh(-closing / 5.0)
    values[1] = math.tanh(right_rate / 3.0)
    values[10] = 1.0 if visible else 0.0
    values[11] = math.tanh(forward / 10.0)
    values[12] = math.tanh(right / 5.0)
    values[23] = gate / 6.0
    return values


def test_non_gate3_actions_are_unchanged_and_reset_state():
    controller = policy.Gate3ProjectedPathPD()
    base = [0.2, -0.3, 0.4, -0.5]
    controller.apply(_observation(), base)
    assert controller.active
    assert controller.apply(_observation(gate=3), base) == base
    assert not controller.active


def test_only_roll_changes_after_bounded_admission():
    controller = policy.Gate3ProjectedPathPD()
    base = [0.2, -0.3, 0.4, -0.5]
    assert controller.apply(_observation(forward=13.0), base) == base
    governed = controller.apply(_observation(forward=10.0), base, dt_s=0.1)
    assert governed[0] == base[0]
    assert governed[2:] == base[2:]
    assert governed[1] != base[1]
    assert controller.active


def test_controller_commands_positive_bank_then_counter_bank():
    controller = policy.Gate3ProjectedPathPD(
        policy.Parameters(max_roll_slew_norm_s=100.0)
    )
    base = [0.0, 0.0, 0.0, 0.0]
    early = controller.apply(
        _observation(forward=10.0, right=-2.0, right_rate=0.0),
        base,
    )
    late = controller.apply(
        _observation(forward=2.0, right=-0.1, right_rate=3.0),
        base,
    )
    assert early[1] > 0.0
    assert late[1] < 0.0


def test_controller_enforces_roll_slew_and_interface_bounds():
    controller = policy.Gate3ProjectedPathPD(
        policy.Parameters(max_roll_slew_norm_s=4.0)
    )
    first = controller.apply(
        _observation(forward=10.0, right=-4.0, right_rate=0.0),
        [0.0, 0.0, 0.0, 0.0],
        dt_s=0.05,
    )
    second = controller.apply(
        _observation(forward=2.0, right=1.0, right_rate=4.0),
        [0.0, 0.0, 0.0, 0.0],
        dt_s=0.05,
    )
    assert abs(second[1] - first[1]) <= 0.200001
    assert -1.0 <= first[1] <= 1.0
    assert -1.0 <= second[1] <= 1.0
