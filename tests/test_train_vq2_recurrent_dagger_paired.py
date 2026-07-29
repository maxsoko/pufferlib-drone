from __future__ import annotations

import inspect
import re

import pytest
import torch

from scripts.train_vq2_recurrent_dagger_paired import (
    PairedTrainConfig,
    paired_objective,
    train,
)


def test_paired_objective_gives_each_source_equal_weight_per_update() -> None:
    result = paired_objective(
        torch.tensor(2.0),
        torch.tensor(3.0),
        torch.tensor(6.0),
        torch.tensor(7.0),
        smoothness_weight=0.1,
    )
    assert result == pytest.approx(0.5 * ((2.0 + 0.3) + (6.0 + 0.7)))


def test_paired_fit_keeps_small_parent_schedule_and_full_history_chunks() -> None:
    config = PairedTrainConfig()
    assert config.epochs == 4
    assert config.sequence_chunk == 64
    assert config.agent_batch_size == 8
    assert config.learning_rate == 1e-4


def test_paired_checkpoint_output_path_cannot_be_shadowed() -> None:
    source = inspect.getsource(train)
    assert not re.search(r"^\s*output,\s*next_state\s*=", source, re.MULTILINE)
    assert "bc_output" in source and "dagger_output" in source
