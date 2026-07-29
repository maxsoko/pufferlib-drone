import subprocess
import sys
from pathlib import Path

import torch

from scripts.audit_vq2_prior_horizon_gradient_conflict import (
    _cosine_matrix,
    _direction_row,
    _weight_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cosine_matrix_exposes_opposed_gradients() -> None:
    matrix = _cosine_matrix(torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64))
    assert matrix == [[1.0, -1.0], [-1.0, 1.0]]


def test_direction_row_detects_common_adam_descent() -> None:
    gradients = torch.ones((5, 3), dtype=torch.float64)
    row = _direction_row(
        gradients, torch.ones(5, dtype=torch.float64), torch.ones(5, dtype=torch.float64)
    )
    assert row["common_adam_descent"] is True
    assert all(value < 0 for value in row["directional_derivative"].values())


def test_weight_candidates_are_fixed() -> None:
    candidates, canonical = _weight_candidates()
    assert canonical == 28
    assert candidates.shape == (8220, 5)
    assert (candidates > 0).all()


def test_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_prior_horizon_gradient_conflict.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n727-report" in completed.stdout
    assert "--progress-every" in completed.stdout
