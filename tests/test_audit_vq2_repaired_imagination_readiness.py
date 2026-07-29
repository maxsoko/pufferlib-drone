import subprocess
import sys
from pathlib import Path

from scripts.audit_vq2_repaired_imagination_readiness import (
    FIXED_HORIZONS,
    _prior_admission,
)


ROOT = Path(__file__).resolve().parents[1]


def _metrics() -> dict[str, object]:
    return {
        str(horizon): {
            "pre_event": {
                "posterior_to_open_prior_kl": {"mean": 0.005},
                "categorical_argmax_agreement": {"mean": 0.9},
                "actor_action_drift": {"rmse": 0.0005, "max_abs": 0.005},
                "posterior_progress": {"mae": 0.25},
                "open_prior_progress": {"mae": 0.30},
                "open_vs_posterior_progress": {"mae": 0.05},
            }
        }
        for horizon in FIXED_HORIZONS
    }


def test_prior_admission_accepts_repaired_metrics() -> None:
    assert all(_prior_admission(_metrics()).values())


def test_prior_admission_rejects_one_bad_horizon() -> None:
    metrics = _metrics()
    metrics["16"]["pre_event"]["open_prior_progress"]["mae"] = 0.36
    gates = _prior_admission(metrics)
    assert not gates["h16_progress_mae_within_0p10"]


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_vq2_repaired_imagination_readiness.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n716-report" in completed.stdout
