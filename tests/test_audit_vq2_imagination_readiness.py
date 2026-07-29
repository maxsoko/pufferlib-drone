import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_vq2_imagination_readiness import (
    _anchor_indices,
    _binary_metrics,
    _regression_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_anchor_indices_exclude_event_endpoint() -> None:
    indices = _anchor_indices(321, 16, burn_in=32, stride=4)
    assert indices[0] == 31
    assert indices[-1] + 16 <= 319
    assert np.all(np.diff(indices) == 4)


def test_regression_metrics_are_exact_for_identity() -> None:
    metrics = _regression_metrics(np.asarray([-1.0, 0.0, 2.0]), np.asarray([-1.0, 0.0, 2.0]))
    assert metrics["count"] == 3
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["correlation"] == pytest.approx(1.0)


def test_binary_metrics_score_perfect_predictions() -> None:
    metrics = _binary_metrics(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
    assert metrics["accuracy"] == 1.0
    assert metrics["mae"] == 0.0
    assert metrics["bce"] < 1e-6


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_vq2_imagination_readiness.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--validation-dataset" in completed.stdout
