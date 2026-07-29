import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.freeze_vq2_event_window_partitions import (
    TEST,
    TRAIN,
    VALIDATION,
    _assign_group_partitions,
)


ROOT = Path(__file__).resolve().parents[1]


def test_direct_cli_help_resolves_repository_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/freeze_vq2_event_window_partitions.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-manifest" in result.stdout


def test_assignment_is_deterministic_and_keeps_groups_whole() -> None:
    vector_step = np.asarray(
        [10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 18, 19, 20, 21],
        dtype=np.int32,
    )
    kwargs = dict(
        seed=679,
        minimum_train_events=6,
        minimum_validation_events=3,
        minimum_test_events=3,
    )
    first = _assign_group_partitions(vector_step, **kwargs)
    second = _assign_group_partitions(vector_step, **kwargs)
    assert np.array_equal(first, second)
    assert np.count_nonzero(first == TRAIN) >= 6
    assert np.count_nonzero(first == VALIDATION) >= 3
    assert np.count_nonzero(first == TEST) >= 3
    for group in np.unique(vector_step):
        assert len(np.unique(first[vector_step == group])) == 1


def test_assignment_fails_closed_on_invalid_or_insufficient_input() -> None:
    with pytest.raises(ValueError, match="integral"):
        _assign_group_partitions(
            np.asarray([1.0, 2.0]),
            seed=1,
            minimum_train_events=1,
            minimum_validation_events=1,
            minimum_test_events=1,
        )
    with pytest.raises(RuntimeError, match="cannot satisfy"):
        _assign_group_partitions(
            np.asarray([1, 2, 3]),
            seed=1,
            minimum_train_events=2,
            minimum_validation_events=1,
            minimum_test_events=1,
        )
    with pytest.raises(RuntimeError, match="leaves only"):
        _assign_group_partitions(
            np.asarray([1, 1, 2, 2, 3, 3, 4, 4]),
            seed=1,
            minimum_train_events=3,
            minimum_validation_events=3,
            minimum_test_events=1,
        )
