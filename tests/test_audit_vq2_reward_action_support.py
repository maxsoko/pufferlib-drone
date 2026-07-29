import numpy as np
import pytest

from scripts.audit_vq2_reward_action_support import (
    _candidate_starts,
    _center_bin_fraction,
    _event_support,
    _summary,
)


def test_candidate_starts_excludes_invalid_and_early_terminal() -> None:
    valid = np.ones((6, 2), dtype=bool)
    continuation = np.ones_like(valid)
    valid[1, 1] = False
    continuation[2, 0] = False
    starts = _candidate_starts(valid, continuation, 3)
    # A terminal continuation is legal at the final element only.
    assert starts.tolist() == [[0, 0], [2, 1], [3, 0], [3, 1]]


def test_event_support_counts_world_window_exposure() -> None:
    valid = np.ones((5, 1), dtype=bool)
    continuation = np.ones_like(valid)
    reward = np.zeros((5, 1), dtype=np.float32)
    reward[3, 0] = 2.0
    report = _event_support(
        valid=valid,
        continuation=continuation,
        reward=reward,
        sequence_length=4,
        context=1,
        event_threshold=1.0,
        sampled_sequences=10,
    )
    assert report["native_event_count"] == 1
    assert report["candidate_sequence_count"] == 2
    assert report["candidate_sequences_with_event"] == 2
    assert report["mean_event_targets_per_sampled_sequence"] == 1.0
    assert report["expected_event_targets_at_training_sample_count"] == 10.0
    assert report["distinct_events_reachable_in_world_window"] == 1


def test_event_in_context_is_not_counted_as_supervised_target() -> None:
    valid = np.ones((4, 1), dtype=bool)
    continuation = np.ones_like(valid)
    reward = np.zeros((4, 1), dtype=np.float32)
    reward[0, 0] = 2.0
    report = _event_support(
        valid=valid,
        continuation=continuation,
        reward=reward,
        sequence_length=4,
        context=1,
        event_threshold=1.0,
        sampled_sequences=1,
    )
    assert report["candidate_sequences_with_event"] == 0
    assert report["distinct_events_reachable_in_world_window"] == 0


def test_summary_rejects_nonfinite_or_empty_values() -> None:
    with pytest.raises(RuntimeError):
        _summary(np.asarray([], dtype=np.float32))
    with pytest.raises(RuntimeError):
        _summary(np.asarray([np.nan], dtype=np.float32))
    report = _summary(np.asarray([-1.0, 1.0], dtype=np.float32))
    assert report["mean"] == 0.0
    assert report["std"] == 1.0


def test_center_bin_fraction_uses_numpy_absolute_value() -> None:
    values = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float32)
    assert _center_bin_fraction(values, 0.1705577) == pytest.approx(3.0 / 5.0)
    with pytest.raises(ValueError):
        _center_bin_fraction(np.asarray([], dtype=np.float32), 0.1705577)
