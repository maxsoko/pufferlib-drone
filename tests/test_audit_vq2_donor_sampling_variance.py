import numpy as np

from scripts.audit_vq2_donor_sampling_variance import (
    _balanced_sampling_counts,
    _complete_progress_steps,
    _current_sampling_counts,
)


def test_current_sampling_counts_are_deterministic_and_bounded():
    first, first_dense = _current_sampling_counts(
        20, 4, 10, seed=7, dense_pool_size=8, dense_batch_size=2
    )
    second, second_dense = _current_sampling_counts(
        20, 4, 10, seed=7, dense_pool_size=8, dense_batch_size=2
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first_dense, second_dense)
    assert first.sum() == 40
    assert first_dense.sum() == 20


def test_balanced_three_epoch_contract_is_exact():
    crop = _balanced_sampling_counts(1904, 16, 357, seed=693)
    dense = _balanced_sampling_counts(128, 8, 357, seed=694)
    assert np.all(crop == 3)
    assert dense.sum() == 357 * 8
    assert dense.max() - dense.min() <= 1


def test_complete_progress_steps_requires_every_gate():
    passing = {
        "step": 350,
        "validation_progress": {
            "correlation": 0.9,
            "mae_m": 0.4,
            "near_plane_mae_m": 0.5,
            "direction_accuracy": 0.8,
        },
        "preservation_gates": {"action": True},
    }
    failing = {
        **passing,
        "step": 400,
        "validation_progress": {
            **passing["validation_progress"],
            "direction_accuracy": 0.7,
        },
    }
    thresholds = {
        "minimum_correlation": 0.8,
        "maximum_mae_m": 0.5,
        "maximum_near_plane_mae_m": 0.6,
        "minimum_direction_accuracy": 0.75,
    }
    assert _complete_progress_steps(
        {"validation_history": [passing, failing]}, thresholds
    ) == [350]
