import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.materialize_vq2_event_window_partitions import (
    TEST,
    TRAIN,
    VALIDATION,
    _slice_aligned_arrays,
    _validated_split_indices,
)


ROOT = Path(__file__).resolve().parents[1]


def test_direct_cli_help_resolves_repository_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/materialize_vq2_event_window_partitions.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--partition-manifest" in result.stdout


def test_manifest_validation_routes_every_source_row_once() -> None:
    source_steps = np.asarray([100, 101, 102, 103], dtype=np.int32)
    source_index = np.asarray([2, 0, 3, 1], dtype=np.int32)
    manifest_steps = source_steps[source_index]
    split = np.asarray([TRAIN, VALIDATION, TEST, TRAIN], dtype=np.uint8)
    indices = _validated_split_indices(
        source_steps, source_index, manifest_steps, split
    )
    assert indices[int(TRAIN)].tolist() == [2, 1]
    assert indices[int(VALIDATION)].tolist() == [0]
    assert indices[int(TEST)].tolist() == [3]


def test_manifest_validation_fails_on_bad_source_mapping_or_split() -> None:
    source_steps = np.asarray([10, 11, 12], dtype=np.int32)
    with pytest.raises(RuntimeError, match="not a permutation"):
        _validated_split_indices(
            source_steps,
            np.asarray([0, 0, 2]),
            np.asarray([10, 10, 12]),
            np.asarray([TRAIN, VALIDATION, TEST]),
        )
    with pytest.raises(RuntimeError, match="does not match"):
        _validated_split_indices(
            source_steps,
            np.asarray([0, 1, 2]),
            np.asarray([10, 99, 12]),
            np.asarray([TRAIN, VALIDATION, TEST]),
        )
    with pytest.raises(RuntimeError, match="unknown split"):
        _validated_split_indices(
            source_steps,
            np.asarray([0, 1, 2]),
            source_steps,
            np.asarray([TRAIN, VALIDATION, 9]),
        )


def test_slice_aligned_arrays_is_exact_and_fail_closed() -> None:
    arrays = {
        "mask": np.arange(24).reshape(4, 2, 3),
        "action": np.arange(8).reshape(4, 2),
    }
    selected = _slice_aligned_arrays(arrays, np.asarray([3, 1]))
    assert selected["mask"].tolist() == arrays["mask"][[3, 1]].tolist()
    assert selected["action"].tolist() == arrays["action"][[3, 1]].tolist()
    with pytest.raises(ValueError, match="does not align"):
        _slice_aligned_arrays(
            {"mask": np.zeros((4, 2)), "action": np.zeros((3, 2))},
            np.asarray([0]),
        )
