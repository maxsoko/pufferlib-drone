import numpy as np
import pytest

from scripts.filter_vq2_event_windows_replay import (
    _filter_event_arrays,
    _select_replay_exact_indices,
)


def test_select_replay_exact_indices_is_ordered_and_fail_closed() -> None:
    selected, eligible = _select_replay_exact_indices(
        np.asarray([1e-7, 2e-4, 2e-7, 3e-7]),
        np.asarray([2e-7, 1e-7, 3e-7, 4e-7]),
        np.asarray([0, 0, 1, 0]),
        target_events=2,
        maximum_deterministic_error=1e-5,
        maximum_logits_error=1e-5,
    )
    assert selected.tolist() == [0, 3]
    assert eligible.tolist() == [True, False, False, True]
    with pytest.raises(RuntimeError, match="only 2 replay-exact events"):
        _select_replay_exact_indices(
            np.asarray([1e-7, 2e-4, 2e-7, 3e-7]),
            np.asarray([2e-7, 1e-7, 3e-7, 4e-7]),
            np.asarray([0, 0, 1, 0]),
            target_events=3,
            maximum_deterministic_error=1e-5,
            maximum_logits_error=1e-5,
        )


def test_filter_event_arrays_preserves_aligned_rows() -> None:
    arrays = {
        "mask": np.arange(24).reshape(4, 2, 3),
        "agent": np.asarray([10, 11, 12, 13]),
    }
    filtered = _filter_event_arrays(arrays, np.asarray([0, 3]))
    assert filtered["mask"].shape == (2, 2, 3)
    assert filtered["agent"].tolist() == [10, 13]
    with pytest.raises(ValueError, match="does not align"):
        _filter_event_arrays(
            {"mask": np.zeros((4, 2)), "agent": np.zeros((3,))},
            np.asarray([0]),
        )
