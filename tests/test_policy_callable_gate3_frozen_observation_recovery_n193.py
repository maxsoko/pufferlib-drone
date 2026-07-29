import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate3_frozen_observation_recovery_n193 as policy


def _observation(*, gate=2, visible=True, forward=2.1, closing=8.5,
                 right=1.0, yaw_error=0.44, yaw=0.0):
    values = [0.0] * 32
    values[0] = math.tanh(-closing / 5.0)
    values[1] = math.tanh(1.5 / 3.0)
    values[2] = math.tanh(-3.2 / 3.0)
    values[6] = math.cos(yaw / 2.0)
    values[9] = math.sin(yaw / 2.0)
    values[10] = 1.0 if visible else 0.0
    values[11] = math.tanh(forward / 10.0)
    values[12] = math.tanh(right / 5.0)
    values[13] = math.tanh(-0.8 / 5.0)
    values[14] = yaw_error / (math.pi / 4.0)
    values[23] = gate / 6.0
    values[24 + gate] = 1.0
    return values


def test_fifth_identical_near_plane_sample_levels_and_scans_only_roll_yaw():
    controller = policy.Gate3FrozenObservationRecovery()
    obs = _observation()
    base = [0.14, -1.0, 0.69, -0.0004]
    for _ in range(4):
        assert controller.apply(obs, base) == base
    governed = controller.apply(obs, base)
    assert governed[0] == base[0]
    assert governed[1] == 0.0
    assert governed[2] == base[2]
    assert math.isclose(governed[3], -0.25 / math.pi, abs_tol=1e-7)
    assert controller.snapshot()["activation_count"] == 1


def test_scan_tracks_current_yaw_while_gate_features_remain_frozen():
    controller = policy.Gate3FrozenObservationRecovery()
    base = [0.14, -1.0, 0.69, -0.0004]
    for _ in range(5):
        controller.apply(_observation(yaw=0.0), base)
    governed = controller.apply(_observation(yaw=-0.1), base)
    assert math.isclose(governed[3], -0.35 / math.pi, abs_tol=1e-7)
    assert controller.snapshot()["active"] is True


def test_fresh_gate_motion_releases_recovery_immediately():
    controller = policy.Gate3FrozenObservationRecovery()
    obs = _observation()
    base = [0.14, -1.0, 0.69, -0.0004]
    for _ in range(5):
        controller.apply(obs, base)
    changed = _observation(forward=2.0)
    assert controller.apply(changed, base) == base
    assert controller.snapshot()["active"] is False


def test_other_gate_invisible_far_or_slow_never_triggers():
    controller = policy.Gate3FrozenObservationRecovery()
    base = [0.14, -1.0, 0.69, -0.0004]
    for obs in (
        _observation(gate=1),
        _observation(visible=False),
        _observation(forward=3.1),
        _observation(closing=0.5),
    ):
        for _ in range(6):
            assert controller.apply(obs, base) == base
