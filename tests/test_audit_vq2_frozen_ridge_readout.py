import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_vq2_frozen_ridge_readout import (
    _fit_sha256,
    _ridge_predict,
    _weighted_standardized_ridge,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_weighted_ridge_recovers_linear_target_deterministically() -> None:
    rng = np.random.default_rng(706)
    feature = rng.normal(size=(256, 5))
    target = feature @ np.asarray([0.5, -1.0, 0.25, 2.0, -0.75]) + 3.0
    weight = rng.uniform(0.5, 2.0, size=len(target))
    first, summary = _weighted_standardized_ridge(
        feature, target, weight, ridge=0.001, minimum_feature_std=1e-6
    )
    second, _ = _weighted_standardized_ridge(
        feature, target, weight, ridge=0.001, minimum_feature_std=1e-6
    )
    prediction = _ridge_predict(first, feature)
    assert np.sqrt(np.mean(np.square(prediction - target))) < 0.01
    assert summary["active_features"] == 5
    assert _fit_sha256(first) == _fit_sha256(second)


def test_fixed_ridge_rejects_nonpositive_regularization() -> None:
    with pytest.raises(ValueError):
        _weighted_standardized_ridge(
            np.ones((2, 1)), np.asarray([0.0, 1.0]), np.ones(2),
            ridge=0.0, minimum_feature_std=1e-6,
        )


def test_direct_cli_excludes_test_optimizer_and_checkpoint_output() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_vq2_frozen_ridge_readout.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--train-dataset" in result.stdout
    assert "--validation-dataset" in result.stdout
    assert "test-dataset" not in result.stdout
    assert "optimizer" not in result.stdout
    assert "output-checkpoint" not in result.stdout
