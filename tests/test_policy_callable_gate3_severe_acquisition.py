import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate3_severe_acquisition as policy


def _observation(*, gate=2, visible=True, forward=25.0, right=-10.0, yaw=-0.4):
    values = np.zeros(32, dtype=np.float32)
    values[6] = 1.0
    values[10] = 1.0 if visible else 0.0
    values[11] = np.float32(math.tanh(forward / 10.0))
    values[12] = np.float32(math.tanh(right / 5.0))
    values[14] = np.float32(yaw / (math.pi / 4.0))
    values[23] = np.float32(gate / 6.0)
    values[24 + gate] = 1.0
    return values


def test_normal_gate3_approach_is_exact_noop():
    controller = policy.Gate3SevereAcquisition()
    base = [0.1, 0.2, 0.3, 0.4]
    assert controller.apply(_observation(right=-5.0), base) == base
    assert controller.snapshot()["activations"] == 0


def test_severe_gate3_uses_bounded_acquisition_command():
    controller = policy.Gate3SevereAcquisition()
    governed = controller.apply(_observation(), [0.8, 1.0, -1.0, 0.0])
    assert governed[:3] == [-0.24, 0.0, 0.0]
    assert math.isclose(governed[3], 0.35 / math.pi, abs_tol=1e-7)
    snapshot = controller.snapshot()
    assert snapshot["active"] is True
    assert snapshot["activations"] == 1
    assert snapshot["steps_active"] == 1


def test_aligned_pose_releases_to_base_exactly():
    controller = policy.Gate3SevereAcquisition()
    controller.apply(_observation(), [0.0] * 4)
    base = [0.1, -0.2, 0.3, -0.4]
    governed = controller.apply(
        _observation(forward=24.0, right=4.0, yaw=0.1), base
    )
    assert governed == base
    assert controller.snapshot()["active"] is False
    assert controller.snapshot()["releases"] == 1


def test_dropout_is_bounded_then_delegates():
    controller = policy.Gate3SevereAcquisition()
    controller.apply(_observation(), [0.0] * 4)
    hidden = _observation(visible=False, forward=0.0, right=0.0, yaw=0.0)
    for _ in range(policy.PARAMETERS.max_dropout_ticks):
        assert controller.apply(hidden, [0.5] * 4)[:3] == [-0.24, 0.0, 0.0]
    assert controller.apply(hidden, [0.5] * 4) == [0.5] * 4
    assert controller.snapshot()["dropout_deactivations"] == 1


def test_other_gate_is_exact_noop_and_deactivates():
    controller = policy.Gate3SevereAcquisition()
    controller.apply(_observation(), [0.0] * 4)
    base = [0.1, 0.2, 0.3, 0.4]
    assert controller.apply(_observation(gate=3), base) == base
    assert controller.snapshot()["active"] is False
