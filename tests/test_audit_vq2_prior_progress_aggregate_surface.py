import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audit_vq2_prior_progress_aggregate_surface import _mean_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_mean_metrics_aggregates_rmse_by_squared_error() -> None:
    rows = [
        {"kl": 1.0, "level": 2.0, "delta": 3.0, "progress_mae": 4.0, "progress_rmse": 3.0},
        {"kl": 3.0, "level": 4.0, "delta": 5.0, "progress_mae": 6.0, "progress_rmse": 4.0},
    ]
    metrics = _mean_metrics(rows)
    assert metrics["kl"] == 2.0
    assert metrics["progress_mae"] == 5.0
    assert metrics["progress_rmse"] == pytest.approx((12.5) ** 0.5)


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_vq2_prior_progress_aggregate_surface.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n722-preregistration" in completed.stdout
