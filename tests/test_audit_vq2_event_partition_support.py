import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_vq2_event_partition_support import (
    _crop_exposure_counts,
    _histogram,
    _plane_distance_m,
)


ROOT = Path(__file__).resolve().parents[1]


def test_direct_cli_help_resolves_repository_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_event_partition_support.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--train-dataset" in result.stdout
    assert "test-dataset" not in result.stdout


def test_crop_exposure_counts_matches_burned_overlapping_crops() -> None:
    exposure = _crop_exposure_counts(
        8, (0, 2, 4), sequence_length=4, burn_in=1
    )
    assert exposure.tolist() == [0, 1, 1, 2, 1, 2, 1, 1]
    with pytest.raises(ValueError, match="outside"):
        _crop_exposure_counts(8, (5,), sequence_length=4, burn_in=1)


def test_plane_distance_and_histogram_are_metric_exact() -> None:
    tail = np.zeros((1, 3, 56), dtype=np.float32)
    tail[0, :, 25] = np.tanh(np.asarray([-1.0, 0.0, 2.0]) / 10.0)
    distance = _plane_distance_m(tail)
    assert np.allclose(distance, [[-1.0, 0.0, 2.0]], atol=1e-6)
    histogram = _histogram(distance, (-np.inf, 0.0, 1.0, np.inf))
    assert [row["count"] for row in histogram] == [1, 1, 1]


def test_plane_distance_rejects_missing_or_nonfinite_target() -> None:
    with pytest.raises(ValueError, match="does not contain"):
        _plane_distance_m(np.zeros((1, 2, 25), dtype=np.float32))
    tail = np.zeros((1, 2, 56), dtype=np.float32)
    tail[0, 0, 25] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        _plane_distance_m(tail)
