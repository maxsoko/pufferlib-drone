import subprocess
import sys
from pathlib import Path

import torch

from scripts.audit_vq2_prior_progress_virtual_surface import (
    _objective_value,
    _virtual_adam_step,
)


ROOT = Path(__file__).resolve().parents[1]


def test_virtual_adam_first_step_matches_sign_normalization() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    _virtual_adam_step(
        [("p", parameter)],
        {"p": torch.tensor([1.0, -1.0])},
        {"p": torch.tensor([2.0, -4.0])},
        1e-3,
        epsilon=1e-12,
    )
    assert torch.allclose(parameter.detach(), torch.tensor([0.999, -0.999]))


def test_objective_value_uses_fixed_weights() -> None:
    metrics = {"kl": 1.0, "level": 2.0, "delta": 3.0}
    assert _objective_value(metrics, (1.0, 0.5, 2.0)) == 8.0


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_vq2_prior_progress_virtual_surface.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n718-preregistration" in completed.stdout
