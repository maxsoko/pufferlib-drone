import numpy as np

from scripts.audit_vq2_broad_visual_association import (
    _metrics,
    _pair_indices,
    _selection_key,
)


def test_pair_indices_require_same_group_and_exact_stride() -> None:
    group = np.asarray([0, 0, 0, 1, 1, 1])
    step = np.asarray([4, 8, 16, 4, 8, 12])
    left, right = _pair_indices(group, step, sample_stride=4)
    assert left.tolist() == [0, 3, 4]
    assert right.tolist() == [1, 4, 5]


def test_broad_metrics_accept_perfect_prediction() -> None:
    target = np.asarray([0.2, 0.0, 3.0, 2.0, 4.0, 3.0])
    phase = np.asarray([0, 0, 1, 1, 2, 2])
    chosen = np.ones(6, dtype=bool)
    result = _metrics(
        target,
        target.copy(),
        phase,
        chosen,
        np.asarray([0, 2, 4]),
        np.asarray([1, 3, 5]),
        near_plane_m=1.0,
        minimum_pair_delta_m=0.02,
    )
    assert result["correlation"] == 1.0
    assert result["mae_m"] == 0.0
    assert result["near_plane_mae_m"] == 0.0
    assert result["direction_accuracy"] == 1.0


def test_selection_prioritizes_validation_correlation() -> None:
    base = {
        "correlation": 0.8,
        "near_plane_mae_m": 0.1,
        "direction_accuracy": 1.0,
        "mae_m": 0.1,
    }
    higher = dict(base, correlation=0.81, near_plane_mae_m=10.0)
    assert _selection_key(higher, 650, "b") > _selection_key(base, 649, "a")
