import numpy as np
import pytest

from scripts.audit_vq2_mask_geometry_capacity import (
    _mask_geometry,
    _pooled_mask,
    _temporal_features,
)


def test_mask_geometry_tracks_translation_and_scale() -> None:
    mask = np.zeros((3, 64, 64), dtype=np.float32)
    mask[0, 24:32, 24:32] = 1.0
    mask[1, 24:32, 36:44] = 1.0
    mask[2, 20:36, 20:36] = 1.0
    feature, names = _mask_geometry(mask)
    center_x = names.index("center_x")
    mass = names.index("mass")
    variance_x = names.index("variance_x")
    assert feature[1, center_x] > feature[0, center_x]
    assert feature[2, mass] > feature[0, mass]
    assert feature[2, variance_x] > feature[0, variance_x]


def test_pooled_mask_preserves_normalized_mean() -> None:
    rng = np.random.default_rng(645)
    mask = rng.random((2, 5, 64, 64), dtype=np.float32)
    pooled = _pooled_mask(mask, 8)
    assert pooled.shape == (2, 5, 8, 8)
    np.testing.assert_allclose(pooled.mean((-2, -1)), mask.mean((-2, -1)), atol=1e-7)
    with pytest.raises(ValueError):
        _pooled_mask(mask, 7)


def test_temporal_features_align_current_then_lags_and_deltas() -> None:
    series = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    feature = _temporal_features(series, (5, 6), depth=3)
    assert feature.shape == (2, 2, 15)
    np.testing.assert_array_equal(feature[0, 0, :9], series[0, [4, 3, 2]].reshape(-1))
    np.testing.assert_array_equal(feature[0, 0, 9:12], series[0, 4] - series[0, 3])
    with pytest.raises(ValueError):
        _temporal_features(series, (2,), depth=3)
