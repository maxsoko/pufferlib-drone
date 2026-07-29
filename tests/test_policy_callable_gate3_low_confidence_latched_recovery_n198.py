import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate3_low_confidence_latched_recovery_n198 as policy


def _observation(
    *, gate=2, visible=True, confidence=0.02, forward=2.5,
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


def test_fifth_weak_confidence_sample_levels_and_scans_only_roll_yaw():
    controller = policy.Gate3LowConfidenceLatchedRecovery()
    observation = _observation()
    base = [0.2, -1.0, 0.75, 0.0]
    for _ in range(4):
        assert controller.apply(observation, base) == base
    governed = controller.apply(observation, base)
    assert governed[0] == base[0]
    assert governed[1] == 0.0
    assert governed[2] == base[2]
    assert governed[3] == -0.25 / math.pi
    assert controller.snapshot()["activation_count"] == 1


def test_recovery_latches_across_positive_confidence_until_transition():
    controller = policy.Gate3LowConfidenceLatchedRecovery()
    base = [0.2, -1.0, 0.75, 0.0]
    for _ in range(5):
        controller.apply(_observation(), base)
    assert controller.apply(_observation(confidence=0.8, forward=20.0), base)[1] == 0.0
    assert controller.snapshot()["active"] is True
    assert controller.apply(_observation(gate=3), base) == base
    assert controller.snapshot()["active"] is False


def test_ineligible_sample_before_trigger_resets_watchdog():
    controller = policy.Gate3LowConfidenceLatchedRecovery()
    base = [0.2, -1.0, 0.75, 0.0]
    for _ in range(4):
        controller.apply(_observation(), base)
    assert controller.apply(_observation(confidence=0.03), base) == base
    for _ in range(4):
        assert controller.apply(_observation(), base) == base
    assert controller.snapshot()["activation_count"] == 0


def test_other_gate_invisible_far_or_slow_never_triggers():
    base = [0.2, -1.0, 0.75, 0.0]
    for observation in (
        _observation(gate=1),
        _observation(visible=False),
        _observation(forward=3.1),
        _observation(closing=0.5),
    ):
        controller = policy.Gate3LowConfidenceLatchedRecovery()
        for _ in range(8):
            assert controller.apply(observation, base) == base
        assert controller.snapshot()["activation_count"] == 0


def test_infer_evaluates_exact_n198_once_before_recovery(monkeypatch):
    observation = _observation()
    events = []
    monkeypatch.setattr(
        policy.n198,
        "infer",
        lambda value: events.append(("n198", value)) or [0.2, -1.0, 0.75, 0.0],
    )
    monkeypatch.setattr(
        policy._RECOVERY,
        "apply",
        lambda value, action: events.append(("recovery", value, action)) or action,
    )
    assert policy.infer(observation) == [0.2, -1.0, 0.75, 0.0]
    assert [event[0] for event in events] == ["n198", "recovery"]
    assert all(event[1] is observation for event in events)
