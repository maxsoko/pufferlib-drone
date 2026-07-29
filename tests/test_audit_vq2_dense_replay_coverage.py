import numpy as np
import pytest

from scripts.audit_vq2_dense_replay_coverage import (
    _chronological_indices,
    _episode_groups,
    _parse_phase_values,
    _stratified_episode_group_split,
    _temporal_candidates,
)


def test_parse_phase_values_is_explicit_and_bounded() -> None:
    assert _parse_phase_values("0") == (0,)
    assert _parse_phase_values("0,2,5") == (0, 2, 5)
    with pytest.raises(ValueError):
        _parse_phase_values("")
    with pytest.raises(ValueError):
        _parse_phase_values("0,0")
    with pytest.raises(ValueError):
        _parse_phase_values("6")


def test_chronological_indices_unwrap_full_ring() -> None:
    np.testing.assert_array_equal(
        _chronological_indices(8, 8, 3),
        np.asarray([3, 4, 5, 6, 7, 0, 1, 2]),
    )
    np.testing.assert_array_equal(_chronological_indices(8, 3, 3), [0, 1, 2])
    with pytest.raises(ValueError):
        _chronological_indices(8, 9, 0)


def test_episode_groups_discard_unknown_leading_fragment() -> None:
    valid = np.asarray([[1], [1], [0], [1], [1], [1], [0], [1]], dtype=bool)
    continuation = np.asarray([[1], [0], [0], [1], [1], [0], [0], [1]], dtype=bool)
    groups, complete = _episode_groups(
        valid, continuation, starts_at_known_boundary=False
    )
    assert groups[:3, 0].tolist() == [-1, -1, -1]
    assert groups[3:6, 0].tolist() == [0, 0, 0]
    assert complete == {0: True, 1: False}


def test_temporal_candidates_do_not_cross_episode_boundaries() -> None:
    groups = np.asarray(
        [[0, 2], [0, 2], [0, 2], [1, 2], [1, 2], [1, 2]], dtype=np.int64
    )
    candidates = _temporal_candidates(groups, depth=3, stride=1)
    assert {tuple(value) for value in candidates.tolist()} == {
        (2, 0),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 0),
        (5, 1),
    }


def test_dense_split_targets_phase_groups_not_phase_rows() -> None:
    groups = np.repeat(np.arange(18), 20)
    phases = np.repeat(np.tile(np.arange(6), 3), 20)
    split = _stratified_episode_group_split(
        groups,
        phases,
        seed=648,
        validation_fraction=0.125,
        test_fraction=0.25,
    )
    for phase in range(6):
        for partition in (0, 1, 2):
            assert np.any((phases == phase) & (split == partition))
    for group in np.unique(groups):
        assert np.unique(split[groups == group]).size == 1
