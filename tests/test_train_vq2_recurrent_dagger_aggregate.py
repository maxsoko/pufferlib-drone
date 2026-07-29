from __future__ import annotations

from scripts.train_vq2_recurrent_dagger_aggregate import (
    AggregateTrainConfig,
    DAGGER_SOURCE_PATTERN,
    aggregate_training_admitted,
    aggregate_validation_score,
)


def _validation(weighted: float, count: int, mse: list[float]) -> dict:
    return {"weighted_mse": weighted, "count": count, "mse": mse}


def test_aggregate_schedule_tracks_dagger_record_ratio() -> None:
    assert DAGGER_SOURCE_PATTERN == ("broad", "next", "next", "next")
    config = AggregateTrainConfig()
    assert config.epochs == 4
    assert config.action_weights == (1.0, 1.0, 4.0, 1.0)
    assert config.loss_action_weights is None
    assert config.prefer_admitted is False
    assert config.learning_rate == 1e-4


def test_aggregate_score_record_weights_dagger_then_pairs_bc() -> None:
    bc = _validation(0.01, 10, [0.01] * 4)
    broad = _validation(0.02, 10, [0.02] * 4)
    next_ = _validation(0.04, 30, [0.04] * 4)
    assert aggregate_validation_score(bc, broad, next_) == 0.0225


def test_aggregate_admission_requires_every_distribution_and_channel() -> None:
    bc = _validation(0.009, 10, [0.001] * 4)
    broad = _validation(0.019, 10, [0.049] * 4)
    next_ = _validation(0.019, 30, [0.049] * 4)
    assert aggregate_training_admitted(bc, broad, next_)
    next_["mse"][2] = 0.051
    assert not aggregate_training_admitted(bc, broad, next_)
