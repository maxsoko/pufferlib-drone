import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.audit_vq2_rotation_invariant_progress_geometry import (
    _householder_rotate,
    _progress_geometry_loss,
    _weighted_covariance_score,
)


ROOT = Path(__file__).resolve().parents[1]


def test_weighted_covariance_score_is_one_for_exact_scalar_embedding() -> None:
    target = torch.asarray([[0.0, 1.0, 2.0, 3.0]])
    feature = torch.stack((target, 2.0 * target), -1)
    weight = torch.ones_like(target)
    score, metrics = _weighted_covariance_score(feature, target, weight)
    assert score.item() == pytest.approx(1.0, abs=1e-7)
    assert metrics["feature_trace_variance"] > 0.0
    assert metrics["target_variance"] > 0.0


def test_progress_geometry_is_rotation_and_target_sign_invariant() -> None:
    generator = torch.Generator().manual_seed(704)
    feature = torch.randn((3, 6, 12), generator=generator)
    target = torch.randn((3, 6), generator=generator)
    level_weight = torch.rand((3, 6), generator=generator) + 0.1
    pair_weight = torch.rand((3, 5), generator=generator) + 0.1
    loss, _ = _progress_geometry_loss(
        feature,
        target,
        level_weight,
        pair_weight,
        level_weight_coefficient=1.0,
        delta_weight_coefficient=1.0,
    )
    rotated, _ = _progress_geometry_loss(
        _householder_rotate(feature),
        target,
        level_weight,
        pair_weight,
        level_weight_coefficient=1.0,
        delta_weight_coefficient=1.0,
    )
    flipped, _ = _progress_geometry_loss(
        feature,
        -target,
        level_weight,
        pair_weight,
        level_weight_coefficient=1.0,
        delta_weight_coefficient=1.0,
    )
    assert rotated.item() == pytest.approx(loss.item(), abs=1e-6)
    assert flipped.item() == pytest.approx(loss.item(), abs=1e-7)


def test_direct_cli_excludes_validation_test_optimizer_and_checkpoint_output() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_vq2_rotation_invariant_progress_geometry.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--train-dataset" in result.stdout
    assert "validation-dataset" not in result.stdout
    assert "test-dataset" not in result.stdout
    assert "optimizer" not in result.stdout
    assert "output-checkpoint" not in result.stdout
