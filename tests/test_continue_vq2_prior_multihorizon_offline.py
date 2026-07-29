import subprocess
import sys
from pathlib import Path

from scripts.audit_vq2_prior_multihorizon_surface import HORIZONS
from scripts.continue_vq2_prior_multihorizon_offline import _metric_parity


ROOT = Path(__file__).resolve().parents[1]


def _rows(value: float) -> dict[str, dict[str, float]]:
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


def test_metric_parity_finds_largest_absolute_difference() -> None:
    expected = _rows(0.2)
    actual = _rows(0.2)
    actual["16"]["progress_mae"] = 0.25
    assert abs(_metric_parity(actual, expected) - 0.05) < 1e-12


def test_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/continue_vq2_prior_multihorizon_offline.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n724-report" in completed.stdout
    assert "--output-checkpoint" in completed.stdout
