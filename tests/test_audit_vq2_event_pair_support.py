import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_vq2_event_pair_support import (
    _crop_pair_exposure_counts,
    _pair_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def test_pair_exposure_counts_only_trained_deltas() -> None:
    exposure = _crop_pair_exposure_counts(
        8, (0, 2, 4), sequence_length=4, burn_in=1
    )
    assert exposure.tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="outside"):
        _crop_pair_exposure_counts(8, (5,), sequence_length=4, burn_in=1)


def test_pair_summary_matches_unique_and_repeated_support() -> None:
    target = np.asarray([[3.0, 2.0, 0.5, 0.0]], dtype=np.float32)
    exposure = np.asarray([0, 1, 2, 1], dtype=np.int64)
    summary = _pair_summary(
        target,
        exposure,
        near_plane_m=1.0,
        minimum_pair_delta_m=0.02,
        near_plane_multiplier=2.0,
    )
    assert summary["unique_pairs"] == 3
    assert summary["crop_pair_exposures"] == 4
    assert summary["near_plane_unique_pairs"] == 2
    assert summary["near_plane_crop_pair_exposures"] == 3
    assert summary["expected_weighted_near_fraction"] == pytest.approx(0.8)


def test_direct_cli_has_no_test_dataset_path() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_event_pair_support.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--sample-support-report" in result.stdout
    assert "test-dataset" not in result.stdout
