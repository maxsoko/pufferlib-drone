import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate3_zero_confidence_recovery_n196 as policy


def _observation(
    *, gate=2, visible=True, confidence=0.0, forward=2.8,
    closing=7.0, right=1.0, yaw_error=0.34,
):
    values = [0.0] * 32
    values[0] = math.tanh(-closing / 5.0)
    values[6] = 1.0
    values[10] = 1.0 if visible else 0.0
    values[11] = math.tanh(forward / 10.0)
    values[12] = math.tanh(right / 5.0)
    values[14] = yaw_error / (math.pi / 4.0)
    values[23] = gate / 6.0
    values[24 + gate] = 1.0
    values[30] = confidence
    return values


def test_fifth_zero_confidence_sample_levels_and_scans_only_roll_yaw():
    controller = policy.Gate3ZeroConfidenceRecovery()
    observation = _observation()
    base = [0.2, -1.0, 0.75, 0.0]
    for _ in range(4):
        assert controller.apply(observation, base) == base
    governed = controller.apply(observation, base)
    assert governed[0] == base[0]
    assert governed[1] == 0.0
    assert governed[2] == base[2]
    assert governed[3] == -0.25 / math.pi
    snapshot = controller.snapshot()
    assert snapshot["activation_count"] == 1
    assert snapshot["active_samples"] == 1


def test_positive_confidence_releases_immediately():
    controller = policy.Gate3ZeroConfidenceRecovery()
    base = [0.2, -1.0, 0.75, 0.0]
    for _ in range(5):
        controller.apply(_observation(), base)
    assert controller.apply(_observation(confidence=0.01), base) == base
    assert controller.snapshot()["active"] is False


def test_other_gate_invisible_far_or_slow_never_activates():
    base = [0.2, -1.0, 0.75, 0.0]
    for observation in (
        _observation(gate=1),
        _observation(visible=False),
        _observation(forward=3.1),
        _observation(closing=0.5),
    ):
        controller = policy.Gate3ZeroConfidenceRecovery()
        for _ in range(8):
            assert controller.apply(observation, base) == base
        assert controller.snapshot()["activation_count"] == 0


def test_infer_evaluates_exact_n196_once_before_recovery(monkeypatch):
    observation = _observation()
    events = []
    monkeypatch.setattr(
        policy.n196,
        "infer",
        lambda value: events.append(("n196", value)) or [0.2, -1.0, 0.75, 0.0],
    )
    monkeypatch.setattr(
        policy._RECOVERY,
        "apply",
        lambda value, action: events.append(("recovery", value, action)) or action,
    )
    assert policy.infer(observation) == [0.2, -1.0, 0.75, 0.0]
    assert [event[0] for event in events] == ["n196", "recovery"]
    assert all(event[1] is observation for event in events)
