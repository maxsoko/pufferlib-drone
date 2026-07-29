import subprocess
import sys
from pathlib import Path

from scripts.audit_vq2_progress_prior_validation import (
    FIXED_HORIZONS,
    _progress_deltas,
)


ROOT = Path(__file__).resolve().parents[1]


def _metrics(value: float) -> dict[str, object]:
    return {
        str(horizon): {
            "pre_event": {
                "open_vs_posterior_progress": {"mae": value},
                "open_prior_progress": {"mae": value + 0.1},
                "posterior_to_open_prior_kl": {"mean": value / 100},
                "actor_action_drift": {"rmse": value / 1000},
            }
        }
        for horizon in FIXED_HORIZONS
    }


def test_progress_deltas_report_improvement() -> None:
    deltas = _progress_deltas(_metrics(0.2), _metrics(0.3))
    assert all(row["open_vs_posterior_mae_delta"] < 0 for row in deltas.values())


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_vq2_progress_prior_validation.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n720-report" in completed.stdout
