import numpy as np
import pytest

from pufferlib.vq2_informed import ENV_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_success_prefix_sequence_capacity import (
    _crop_batch,
    _crop_index,
    _metrics,
)


def test_crop_index_keeps_event_crops_adjacent() -> None:
    event, start = _crop_index(2, (0, 4, 8))
    assert event.tolist() == [0, 0, 0, 1, 1, 1]
    assert start.tolist() == [0, 4, 8, 0, 4, 8]


def test_crop_batch_is_chronological_and_legal() -> None:
    mask = np.zeros((2, 8, MASK_SIZE), dtype=np.uint8)
    tail = np.zeros((2, 8, ENV_OBS_SIZE - MASK_SIZE), dtype=np.float32)
    for step in range(8):
        mask[:, step, 0] = step
        tail[:, step, 0] = step
        tail[:, step, 25] = np.tanh(step / 10.0)
    selected_mask, selected_tail, target = _crop_batch(
        mask,
        tail,
        np.asarray([1]),
        np.asarray([2]),
        sequence_length=4,
    )
    assert selected_mask[0, :, 0].tolist() == [2, 3, 4, 5]
    assert selected_tail[0, :, 0].tolist() == [2, 3, 4, 5]
    np.testing.assert_allclose(target[0], [2, 3, 4, 5], atol=1e-5)
    with pytest.raises(ValueError):
        _crop_batch(mask, tail, np.asarray([0]), np.asarray([6]), sequence_length=4)


def test_success_prefix_metrics_accept_perfect_decrease() -> None:
    target = np.asarray([[4.0, 3.0, 2.0, 1.0], [3.5, 2.5, 1.5, 0.5]])
    result = _metrics(
        target,
        target.copy(),
        evaluation_indices=np.arange(4),
        near_plane_m=1.0,
        minimum_pair_delta_m=0.02,
    )
    assert result["correlation"] == pytest.approx(1.0)
    assert result["mae_m"] == 0.0
    assert result["near_plane_mae_m"] == 0.0
    assert result["direction_accuracy"] == 1.0
    assert result["strictly_decreasing_crop_fraction"] == 1.0
