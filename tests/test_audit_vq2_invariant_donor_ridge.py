import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.audit_vq2_invariant_donor_ridge import (
    _covered_history_exact,
    _fixed_admission_gates,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_admission_gates_accept_exact_boundaries() -> None:
    assert all(
        _fixed_admission_gates(
            {
                "correlation": 0.80,
                "mae_m": 0.50,
                "near_plane_mae_m": 0.60,
                "direction_accuracy": 0.75,
            }
        ).values()
    )


def test_fixed_admission_gates_reject_each_failure() -> None:
    baseline = {
        "correlation": 0.80,
        "mae_m": 0.50,
        "near_plane_mae_m": 0.60,
        "direction_accuracy": 0.75,
    }
    failures = {
        "correlation": 0.799,
        "mae_m": 0.501,
        "near_plane_mae_m": 0.601,
        "direction_accuracy": 0.749,
    }
    for name, value in failures.items():
        candidate = dict(baseline)
        candidate[name] = value
        gates = _fixed_admission_gates(candidate)
        assert sum(not passed for passed in gates.values()) == 1


def test_covered_history_gate_returns_native_json_boolean() -> None:
    covered = np.arange(32, 320, dtype=np.int64)
    value = _covered_history_exact(covered)
    assert value is True
    assert type(value) is bool


def test_direct_cli_excludes_test_optimizer_and_checkpoint_output() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_vq2_invariant_donor_ridge.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--train-dataset" in result.stdout
    assert "--validation-dataset" in result.stdout
    assert "--n707-report" in result.stdout
    assert "test-dataset" not in result.stdout
    assert "optimizer" not in result.stdout
    assert "output-checkpoint" not in result.stdout
