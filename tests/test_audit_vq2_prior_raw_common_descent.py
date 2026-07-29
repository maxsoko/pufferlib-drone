import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.audit_vq2_prior_raw_common_descent import (
    _minimum_norm_weights,
    _raw_direction_row,
)


ROOT = Path(__file__).resolve().parents[1]


def test_minimum_norm_convex_combination_is_common_descent() -> None:
    gradients = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-0.5, 0.5]],
        dtype=np.float64,
    )
    gram = gradients @ gradients.T
    weights, _iterations, _gap = _minimum_norm_weights(gram)
    row = _raw_direction_row(gram, weights)
    assert row["common_raw_descent"] is True
    assert abs(float(weights.sum()) - 1.0) < 1e-12


def test_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_prior_raw_common_descent.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n728-report" in completed.stdout
