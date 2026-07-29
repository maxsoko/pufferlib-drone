import subprocess
import sys
from pathlib import Path

import torch

from scripts.audit_vq2_prior_raw_common_descent_surface import (
    _apply_unit_step,
    _baseline_parity,
)


ROOT = Path(__file__).resolve().parents[1]


def test_apply_unit_step_maps_flat_direction_to_parameters() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    second = torch.nn.Parameter(torch.tensor([3.0]))
    named = [("a", first), ("b", second)]
    source = {"a": first.detach().clone(), "b": second.detach().clone()}
    _apply_unit_step(named, source, torch.tensor([0.5, -0.5, 1.0]), 0.1)
    assert torch.allclose(first, torch.tensor([0.95, 2.05]))
    assert torch.allclose(second, torch.tensor([2.9]))


def test_baseline_parity_finds_largest_difference() -> None:
    actual = {"1": {"x": 1.0}, "2": {"x": 1.0}, "4": {"x": 1.0}, "8": {"x": 1.0}, "16": {"x": 1.0}}
    expected = {key: dict(value) for key, value in actual.items()}
    expected["16"]["x"] = 1.25
    assert _baseline_parity(actual, expected) == 0.25


def test_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_prior_raw_common_descent_surface.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n729-report" in completed.stdout
    assert "--progress-every" in completed.stdout
