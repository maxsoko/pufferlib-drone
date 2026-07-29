import numpy as np
import pytest
import torch

from scripts.audit_vq2_probe_gradient_alignment import (
    _gradient_alignment_summary,
    _parse_probe_seeds,
    _target_normalization,
)


def test_parse_probe_seeds_requires_distinct_values() -> None:
    assert _parse_probe_seeds("10,11,12") == (10, 11, 12)
    with pytest.raises(ValueError):
        _parse_probe_seeds("10,10")


def test_target_normalization_uses_inverse_crop_exposure() -> None:
    target = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    starts = np.asarray([0, 1])
    exposure = np.asarray([1, 2, 1])
    mean, std = _target_normalization(target, starts, exposure, burn_in=0)
    values = np.asarray([0.0, 2.0, 2.0, 4.0])
    weights = np.asarray([1.0, 0.5, 0.5, 1.0])
    expected_mean = float(np.dot(values, weights) / weights.sum())
    expected_std = float(
        np.sqrt(np.dot(np.square(values - expected_mean), weights) / weights.sum())
    )
    assert mean == pytest.approx(expected_mean)
    assert std == pytest.approx(expected_std)


def test_gradient_alignment_summary_detects_shared_direction() -> None:
    rows = []
    for scale in (1.0, 2.0, 3.0):
        rows.append(
            {
                "encoder": torch.tensor([scale, 2.0 * scale]),
                "sequence": torch.tensor([-scale, -2.0 * scale]),
                "posterior": torch.tensor([0.5 * scale]),
            }
        )
    summary = _gradient_alignment_summary(rows, top_coordinates=5)
    for name in ("encoder", "sequence", "posterior", "overall"):
        assert summary[name]["pairwise_cosine_median"] == pytest.approx(1.0)
        assert summary[name]["pairwise_sign_agreement_median"] == pytest.approx(1.0)
        assert summary[name]["positive_cosine_to_ensemble"] == 3
        assert summary[name]["all_finite"]
