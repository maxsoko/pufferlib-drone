import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import policy_callable_gate2_frozen_observation_dropout_n194 as policy


def _observation(*, gate=1, visible=True, forward=4.2, closing=4.3,
                 right=0.82):
    values = np.zeros(32, dtype=np.float32)
    values[0] = math.tanh(-closing / 5.0)
    values[1] = math.tanh(-0.69 / 3.0)
    values[2] = math.tanh(-3.53 / 3.0)
    values[6] = 1.0
    values[10] = 1.0 if visible else 0.0
    values[11] = math.tanh(forward / 10.0)
    values[12] = math.tanh(right / 5.0)
    values[13] = math.tanh(-1.35 / 5.0)
    values[14] = 0.19 / (math.pi / 4.0)
    values[23] = gate / 6.0
    values[24 + gate] = 1.0
    return values


def test_eighth_frozen_gate2_sample_changes_only_visibility():
    sanitizer = policy.Gate2FrozenObservationDropout()
    observation = _observation()
    for _ in range(7):
        np.testing.assert_array_equal(sanitizer.sanitize(observation), observation)
    sanitized = sanitizer.sanitize(observation)
    expected = observation.copy()
    expected[10] = 0.0
    np.testing.assert_array_equal(sanitized, expected)
    assert sanitizer.snapshot()["activation_count"] == 1
    assert sanitizer.snapshot()["sanitized_samples"] == 1


def test_fresh_gate_motion_releases_dropout_immediately():
    sanitizer = policy.Gate2FrozenObservationDropout()
    observation = _observation()
    for _ in range(8):
        sanitizer.sanitize(observation)
    fresh = _observation(forward=4.1)
    np.testing.assert_array_equal(sanitizer.sanitize(fresh), fresh)
    assert sanitizer.snapshot()["active"] is False


def test_other_gate_invisible_far_or_slow_never_sanitizes():
    sanitizer = policy.Gate2FrozenObservationDropout()
    for observation in (
        _observation(gate=2),
        _observation(visible=False),
        _observation(forward=6.1),
        _observation(closing=0.5),
    ):
        for _ in range(6):
            np.testing.assert_array_equal(sanitizer.sanitize(observation), observation)


def test_reset_clears_watchdog_and_counters():
    sanitizer = policy.Gate2FrozenObservationDropout()
    observation = _observation()
    for _ in range(8):
        sanitizer.sanitize(observation)
    sanitizer.reset()
    snapshot = sanitizer.snapshot()
    assert snapshot["active"] is False
    assert snapshot["activation_count"] == 0
    assert snapshot["active_samples"] == 0


def test_infer_passes_sanitized_dropout_to_exact_n194(monkeypatch):
    monkeypatch.setattr(policy.n194, "reset", lambda: None)
    policy.reset()
    seen_visibility = []

    def fake_infer(observation):
        seen_visibility.append(float(observation[10]))
        return [0.1, 0.2, 0.3, 0.4]

    monkeypatch.setattr(policy.n194, "infer", fake_infer)
    observation = _observation()
    for _ in range(8):
        assert policy.infer(observation) == [0.1, 0.2, 0.3, 0.4]
    assert seen_visibility == [1.0] * 7 + [0.0]
