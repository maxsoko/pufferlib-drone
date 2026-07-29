import numpy as np
import torch

from scripts.audit_vq2_nonlinear_visual_association import (
    PROBE_SPECS,
    VisualAssociationProbe,
    _partition_result,
    _selection_key,
    _temporal_masks,
)


def test_temporal_masks_are_current_first_without_variable_prefix() -> None:
    mask = np.arange(2 * 8 * 64 * 64, dtype=np.float32).reshape(2, 8, 64, 64)
    temporal = _temporal_masks(mask, (5, 6), depth=4)
    assert temporal.shape == (2, 2, 4, 64, 64)
    np.testing.assert_array_equal(temporal[0, 0], mask[0, [4, 3, 2, 1]])


def test_partition_result_accepts_perfect_decreasing_prediction() -> None:
    target = np.asarray([[5.0, 4.0, 3.0], [4.0, 3.0, 2.0]])
    result = _partition_result(target, target.copy(), np.asarray([True, True]))
    assert result["correlation"] == 1.0
    assert result["mae_in_true_step_units"] == 0.0
    assert result["strictly_decreasing_window_fraction"] == 1.0
    assert result["final_is_minimum_window_fraction"] == 1.0


def test_probe_variants_emit_one_scalar_per_sample() -> None:
    mask = torch.zeros(3, 4, 64, 64)
    tail = torch.zeros(3, 22)
    boundary = torch.zeros(3, 256)
    for spec in PROBE_SPECS.values():
        probe = VisualAssociationProbe(
            temporal_depth=4,
            legal_tail_size=22,
            boundary_size=256,
            spec=spec,
        )
        assert probe(mask, tail, boundary).shape == (3,)


def test_selection_key_prefers_correlation_before_rank() -> None:
    base = {
        "correlation": 0.8,
        "final_is_minimum_window_fraction": 1.0,
        "strictly_decreasing_window_fraction": 1.0,
        "mae_in_true_step_units": 1.0,
    }
    higher = dict(base, correlation=0.81, final_is_minimum_window_fraction=0.0)
    assert _selection_key(higher, seed=647, name="b") > _selection_key(
        base, seed=646, name="a"
    )
