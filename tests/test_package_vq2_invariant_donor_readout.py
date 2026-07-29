import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_vq2_frozen_ridge_readout import (
    _fit_sha256,
    _weighted_standardized_ridge,
)
from scripts.package_vq2_invariant_donor_readout import (
    _fit_auxiliary,
    _fit_from_auxiliary,
    _metric_parity_error,
    _strict_action_gates,
    _strict_progress_gates,
)


ROOT = Path(__file__).resolve().parents[1]


def test_strict_gates_accept_exact_boundaries() -> None:
    assert all(
        _strict_progress_gates(
            {
                "correlation": 0.8,
                "mae_m": 0.5,
                "near_plane_mae_m": 0.5,
                "direction_accuracy": 0.75,
            }
        ).values()
    )
    assert all(
        _strict_action_gates(
            {"rmse": 0.001, "max_abs": 0.01},
            {"rmse": 0.001, "max_abs": 0.01},
        ).values()
    )


def test_auxiliary_round_trip_preserves_fit_hash() -> None:
    rng = np.random.default_rng(711)
    feature = rng.normal(size=(128, 7))
    target = feature @ rng.normal(size=7)
    fit, _ = _weighted_standardized_ridge(
        feature, target, np.ones(len(target)),
        ridge=0.001, minimum_feature_std=1e-6,
    )
    fit_hash = _fit_sha256(fit)
    auxiliary = _fit_auxiliary(
        fit,
        fit_sha256=fit_hash,
        covered=np.arange(32, 64, dtype=np.int64),
        source_hashes={"synthetic": "test"},
    )
    assert _fit_sha256(_fit_from_auxiliary(auxiliary)) == fit_hash


def test_metric_parity_uses_all_admission_diagnostics() -> None:
    baseline = {
        "correlation": 0.9,
        "mae_m": 0.3,
        "near_plane_mae_m": 0.4,
        "direction_accuracy": 0.8,
        "rmse_m": 0.35,
        "near_plane_correlation": 0.2,
        "strictly_decreasing_crop_fraction": 0.4,
        "final_is_minimum_crop_fraction": 0.7,
    }
    changed = dict(baseline)
    changed["near_plane_correlation"] += 0.0125
    assert _metric_parity_error(changed, baseline) == pytest.approx(0.0125)


def test_direct_cli_has_no_test_or_optimizer_surface() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/package_vq2_invariant_donor_readout.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--train-dataset" in result.stdout
    assert "--validation-dataset" in result.stdout
    assert "--output-checkpoint" in result.stdout
    assert "test-dataset" not in result.stdout
    assert "optimizer" not in result.stdout
    assert "actor-steps" not in result.stdout
    assert "probe-steps" not in result.stdout
