import subprocess
import sys
from pathlib import Path

from scripts.audit_vq2_prior_multihorizon_aggregate_surface import (
    HORIZONS,
    _mean_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(value: float) -> dict[str, dict[str, float]]:
    return {
        str(horizon): {
            "loss": value,
            "progress_mae": value,
            "progress_rmse": value,
            "progress_bias": value,
            "kl": value,
            "action_rmse": value,
            "action_max_abs": value,
        }
        for horizon in HORIZONS
    }


def test_mean_rows_uses_rms_and_max_for_error_aggregates() -> None:
    mean = _mean_rows([_row(0.0), _row(2.0)])
    assert mean["1"]["progress_mae"] == 1.0
    assert abs(mean["1"]["progress_rmse"] - 2**0.5) < 1e-12
    assert abs(mean["1"]["action_rmse"] - 2**0.5) < 1e-12
    assert mean["1"]["action_max_abs"] == 2.0


def test_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_prior_multihorizon_aggregate_surface.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n726-report" in completed.stdout
    assert "--progress-every" in completed.stdout
