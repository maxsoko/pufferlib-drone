import subprocess
import sys
from pathlib import Path

import torch

from scripts.continue_vq2_prior_progress_offline import _ridge_predict_torch


ROOT = Path(__file__).resolve().parents[1]


def test_torch_ridge_predicts_affine_standardized_feature() -> None:
    ridge = {
        "mean": torch.tensor([1.0, 2.0]),
        "std": torch.tensor([2.0, 4.0]),
        "active": torch.tensor([True, True]),
        "coefficient": torch.tensor([2.0, -1.0, 0.5]),
        "target_mean": torch.tensor(3.0),
        "target_std": torch.tensor(2.0),
    }
    prediction = _ridge_predict_torch(torch.tensor([[3.0, 6.0]]), ridge)
    assert torch.equal(prediction, torch.tensor([6.0]))


def test_torch_ridge_backpropagates_to_feature() -> None:
    feature = torch.tensor([[0.2, 0.4]], requires_grad=True)
    ridge = {
        "mean": torch.zeros(2),
        "std": torch.ones(2),
        "active": torch.tensor([True, True]),
        "coefficient": torch.tensor([1.0, 2.0, 0.0]),
        "target_mean": torch.tensor(0.0),
        "target_std": torch.tensor(1.0),
    }
    _ridge_predict_torch(feature, ridge).sum().backward()
    assert torch.equal(feature.grad, torch.tensor([[1.0, 2.0]]))


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/continue_vq2_prior_progress_offline.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--n717-report" in completed.stdout
    assert "--n719-report" in completed.stdout
