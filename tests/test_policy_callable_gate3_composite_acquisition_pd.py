import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate3_composite_acquisition_pd as policy


def _observation(
    *,
    gate=2,
    visible=True,
    forward=10.0,
    right=-2.0,
    yaw=-0.1,
    closing=8.0,
    right_rate=0.0,
):
    values = [0.0] * 32
    values[0] = math.tanh(-closing / 5.0)
    values[1] = math.tanh(right_rate / 3.0)
    values[6] = 1.0
    values[10] = 1.0 if visible else 0.0
    values[11] = math.tanh(forward / 10.0)
    values[12] = math.tanh(right / 5.0)
    values[14] = yaw / (math.pi / 4.0)
    values[23] = gate / 6.0
    values[24 + gate] = 1.0
    return values


def test_other_gates_delegate_exactly_and_reset_both_branches():
    controller = policy.Gate3CompositeAcquisitionPD()
    controller.apply(_observation(), [0.0] * 4)
    base = [0.1, -0.2, 0.3, -0.4]
    assert controller.apply(_observation(gate=3), base) == base
    snapshot = controller.snapshot()
    assert snapshot["branch"] == "base"
    assert snapshot["severe"]["active"] is False
    assert snapshot["projected"]["active"] is False


def test_severe_branch_has_precedence_and_exact_n188_command():
    controller = policy.Gate3CompositeAcquisitionPD()
    governed = controller.apply(
        _observation(forward=28.0, right=-16.0, yaw=-0.5),
        [0.8, 1.0, -1.0, 0.0],
    )
    assert governed[:3] == [-0.24, 0.0, 0.0]
    assert math.isclose(governed[3], 0.35 / math.pi, abs_tol=1e-7)
    snapshot = controller.snapshot()
    assert snapshot["branch"] == "severe"
    assert snapshot["severe"]["activations"] == 1
    assert snapshot["projected"]["active"] is False


def test_normal_close_approach_uses_pd_and_changes_only_roll():
    controller = policy.Gate3CompositeAcquisitionPD()
    base = [0.2, -0.3, 0.4, -0.5]
    governed = controller.apply(_observation(), base, dt_s=0.1)
    assert governed[0] == base[0]
    assert governed[2:] == base[2:]
    assert governed[1] != base[1]
    snapshot = controller.snapshot()
    assert snapshot["branch"] == "projected"
    assert snapshot["severe"]["activations"] == 0
    assert snapshot["projected"]["active"] is True


def test_severe_release_defers_pd_until_next_sample():
    controller = policy.Gate3CompositeAcquisitionPD()
    controller.apply(
        _observation(forward=28.0, right=-16.0, yaw=-0.5), [0.0] * 4
    )
    base = [0.1, -0.2, 0.3, -0.4]
    released = controller.apply(
        _observation(forward=10.0, right=4.0, yaw=0.1), base
    )
    assert released == base
    assert controller.snapshot()["branch"] == "severe_release"
    assert controller.snapshot()["projected"]["active"] is False
    governed = controller.apply(
        _observation(forward=9.5, right=3.5, yaw=0.08), base
    )
    assert controller.snapshot()["branch"] == "projected"
    assert governed[0] == base[0]
    assert governed[2:] == base[2:]


def test_visibility_loss_drops_pd_hold_and_returns_base():
    controller = policy.Gate3CompositeAcquisitionPD()
    controller.apply(_observation(), [0.0] * 4)
    assert controller.snapshot()["projected"]["active"] is True
    base = [0.2, 0.3, 0.4, 0.5]
    assert controller.apply(_observation(visible=False), base) == base
    snapshot = controller.snapshot()
    assert snapshot["projected"]["active"] is False
    assert snapshot["projected_visibility_resets"] == 1
