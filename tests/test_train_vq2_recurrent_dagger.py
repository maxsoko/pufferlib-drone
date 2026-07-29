from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

from scripts.train_vq2_recurrent_dagger import (
    DaggerTrainConfig,
    aggregate_validation_score,
    source_schedule,
    train,
    train_source_pass,
)


def test_record_balanced_schedule_keeps_one_oracle_anchor() -> None:
    config = DaggerTrainConfig()
    schedule = source_schedule(config, np.random.default_rng(config.seed))
    assert schedule.count("bc") == 1
    assert schedule.count("dagger") == 44
    # 44 * roughly 8k on-policy train records balances one 353k BC pass.
    assert config.dagger_passes_per_epoch == 44


def test_aggregate_selection_weights_both_disjoint_validations_equally() -> None:
    assert aggregate_validation_score(
        {"weighted_mse": 0.02}, {"weighted_mse": 0.04}
    ) == pytest.approx(0.03)
    assert aggregate_validation_score(
        {"weighted_mse": float("nan")}, {"weighted_mse": 0.04}
    ) == float("inf")


def test_dagger_checkpoint_path_cannot_be_shadowed_by_actor_output() -> None:
    source = inspect.getsource(train)
    assert not re.search(r"^\s*output,\s*next_state\s*=", source, re.MULTILINE)
    assert "actor_output" in inspect.getsource(train_source_pass)
