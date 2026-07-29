import subprocess
import sys
from pathlib import Path

from scripts.audit_vq2_imagination_readiness import FIXED_HORIZONS
from scripts.audit_vq2_multihorizon_prior_validation import _improvement_gates


ROOT = Path(__file__).resolve().parents[1]


def test_improvement_gates_require_each_horizon() -> None:
    deltas = {
        str(horizon): {"open_vs_posterior_mae_delta": -0.01}
        for horizon in FIXED_HORIZONS
    }
    assert all(_improvement_gates(deltas, suffix="baseline").values())
    deltas["16"]["open_vs_posterior_mae_delta"] = 0.0
    assert not all(_improvement_gates(deltas, suffix="baseline").values())


def test_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_multihorizon_prior_validation.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n725-report" in completed.stdout
    assert "--validation-dataset" in completed.stdout
