from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
import torch

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, MASK_SIZE
from scripts.train_vq2_recurrent_bc import (
    ACTION_SIZE,
    actor_agent_split,
    reconstruct_legal_batch,
    temporal_smoothness,
    train,
    weighted_action_mse,
)


def test_course_split_is_disjoint_and_deterministic() -> None:
    train, validation = actor_agent_split(64, 8)
    assert train.tolist() == list(range(56))
    assert validation.tolist() == list(range(56, 64))
    assert not np.intersect1d(train, validation).size
    with pytest.raises(ValueError, match="nonempty"):
        actor_agent_split(64, 64)


def test_checkpoint_output_path_is_not_shadowed_by_actor_output() -> None:
    source = inspect.getsource(train)
    assert not re.search(r"^\s*output,\s*next_state\s*=", source, re.MULTILINE)
    assert "actor_output, next_state = model.forward_sequence" in source


def test_reconstruction_decodes_only_mask_and_legal_tail() -> None:
    mask = np.zeros((3, 2, MASK_SIZE), dtype=np.uint8)
    mask[1, 0, 5] = 255
    tail = np.arange(3 * 2 * (LEGAL_OBS_SIZE - MASK_SIZE), dtype=np.float32)
    tail = tail.reshape(3, 2, LEGAL_OBS_SIZE - MASK_SIZE)
    legal = reconstruct_legal_batch(mask, tail, device=torch.device("cpu"))
    assert legal.shape == (2, 3, LEGAL_OBS_SIZE)
    assert legal[0, 1, 5] == 1.0
    assert torch.equal(
        legal[1, 2, MASK_SIZE:], torch.from_numpy(tail[2, 1])
    )


def test_weighted_mse_masks_padding_and_covers_all_four_channels() -> None:
    prediction = torch.zeros(2, 2, ACTION_SIZE)
    target = torch.ones_like(prediction)
    target[1, 1] = 1000.0
    valid = torch.tensor([[True, True], [True, False]])
    weights = torch.tensor([1.0, 1.0, 4.0, 1.0])
    loss, channel = weighted_action_mse(prediction, target, valid, weights)
    assert torch.equal(channel, torch.ones(ACTION_SIZE))
    assert loss == 1.0


def test_temporal_smoothness_includes_chunk_boundary_only_when_valid() -> None:
    prediction = torch.tensor([[[1.0] * 4, [3.0] * 4]])
    valid = torch.tensor([[True, True]])
    smooth = temporal_smoothness(
        prediction,
        valid,
        previous_prediction=torch.zeros(1, 4),
        previous_valid=torch.tensor([True]),
    )
    assert smooth == pytest.approx((1.0 + 4.0) / 2.0)
    masked = temporal_smoothness(
        prediction[:, :1],
        valid[:, :1],
        previous_prediction=torch.zeros(1, 4),
        previous_valid=torch.tensor([False]),
    )
    assert masked == 0.0
