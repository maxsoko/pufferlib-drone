import numpy as np

from scripts.audit_vq2_plane_representation_capacity import (
    _fit_continuous_ridge,
    _passes_capacity,
    _regression_result,
)


def test_continuous_ridge_recovers_heldout_linear_progress():
    rng = np.random.default_rng(632)
    events = 24
    steps = 5
    base = rng.normal(size=(events, 1))
    target = base - np.arange(steps)[None, :] * 0.1
    features = np.stack((target, target**2), -1)
    split = np.asarray([0] * 14 + [1] * 5 + [2] * 5, dtype=np.int8)
    prediction, fit = _fit_continuous_ridge(
        features, target, split, ridge_values=(0.0001, 0.01, 1.0)
    )
    result = _regression_result(target, prediction, split)
    assert fit["selected_ridge"] in (0.0001, 0.01, 1.0)
    assert result["correlation"] > 0.99
    assert result["final_is_minimum_window_fraction"] == 1.0


def test_capacity_gate_rejects_constant_near_zero_prediction():
    result = {
        "correlation": 0.0,
        "mae_in_true_step_units": 1.0,
        "strictly_decreasing_window_fraction": 0.0,
        "final_is_minimum_window_fraction": 0.0,
    }
    assert not _passes_capacity(result)
