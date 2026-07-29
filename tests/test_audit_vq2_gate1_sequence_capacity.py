import numpy as np
import pytest
import torch

from pufferlib.vq2_informed import ENV_OBS_SIZE, MASK_SIZE
from scripts.audit_vq2_gate1_sequence_capacity import (
    PROBE_SPECS,
    RecurrentVisualProbe,
    _gather_sequences,
    _trajectory_metrics,
)


def test_gather_sequences_is_chronological_and_legal() -> None:
    time, agents = 6, 2
    mask = np.zeros((time, agents, MASK_SIZE), dtype=np.uint8)
    tail = np.zeros((time, agents, ENV_OBS_SIZE - MASK_SIZE), dtype=np.float16)
    for step in range(time):
        mask[step, :, 0] = step
        tail[step, :, 0] = step
        tail[step, :, 25] = np.tanh(step / 10.0)
    arrays = {"mask": mask, "tail": tail}
    gathered_mask, gathered_tail, target = _gather_sequences(
        arrays,
        np.arange(time),
        np.asarray([4]),
        np.asarray([1]),
        sequence_length=3,
    )
    assert gathered_mask[0, :, 0].tolist() == [2, 3, 4]
    assert gathered_tail[0, :, 0].tolist() == [2, 3, 4]
    np.testing.assert_allclose(target[0], [2, 3, 4], atol=1e-2)
    with pytest.raises(ValueError):
        _gather_sequences(
            arrays, np.arange(time), np.asarray([1]), np.asarray([0]), sequence_length=3
        )


def test_recurrent_probe_shapes_for_all_legal_specs() -> None:
    mask = torch.zeros(2, 5, 64, 64)
    tail = torch.zeros(2, 5, 22)
    for spec in PROBE_SPECS.values():
        probe = RecurrentVisualProbe(legal_tail_size=22, hidden_size=16, spec=spec)
        assert probe(mask, tail).shape == (2, 5)


def test_trajectory_metrics_accept_perfect_prediction() -> None:
    target = np.asarray([[3.0, 2.0, 1.0], [2.0, 1.5, 1.0]], dtype=np.float32)
    result = _trajectory_metrics(target, target.copy())
    assert result["correlation"] == pytest.approx(1.0)
    assert result["mae_m"] == 0.0
    assert result["delta_mae_m"] == 0.0
