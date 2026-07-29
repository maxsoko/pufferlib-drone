import argparse
import subprocess
import sys
from pathlib import Path

from scripts.audit_vq2_donor_selector import _select_candidate


ROOT = Path(__file__).resolve().parents[1]


def _candidate(step: int, correlation: float, mae: float) -> dict:
    return {
        "step": step,
        "validation_progress": {
            "correlation": correlation,
            "mae_m": mae,
            "near_plane_mae_m": 0.4,
            "direction_accuracy": 0.8,
        },
        "preservation_gates": {"action": True},
    }


def test_selector_prefers_complete_gate_pass_over_higher_correlation() -> None:
    args = argparse.Namespace(
        minimum_validation_correlation=0.8,
        maximum_validation_mae_m=0.5,
        maximum_validation_near_plane_mae_m=0.6,
        minimum_validation_direction_accuracy=0.75,
    )
    selected = _select_candidate(
        [_candidate(350, 0.90, 0.38), _candidate(400, 0.92, 0.51)], args
    )
    assert selected["step"] == 350


def test_direct_cli_has_no_test_or_checkpoint_argument() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_vq2_donor_selector.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--source-report" in result.stdout
    assert "test-dataset" not in result.stdout
    assert "checkpoint" not in result.stdout
