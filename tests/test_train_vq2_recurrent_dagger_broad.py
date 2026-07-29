from __future__ import annotations

from scripts.train_vq2_recurrent_dagger_broad import (
    BroadTrainConfig,
    broad_training_admitted,
)


def _validation(weighted: float, mse: list[float]) -> dict:
    return {"weighted_mse": weighted, "mse": mse}


def test_broad_training_contract_keeps_simple_paired_fit() -> None:
    config = BroadTrainConfig()
    assert config.epochs == 4
    assert config.action_weights == (1.0, 1.0, 4.0, 1.0)
    assert config.bc_validation_agents == 8
    assert config.broad_validation_agents == 64
    assert config.learning_rate == 1e-4


def test_broad_admission_requires_both_sources_and_every_channel() -> None:
    bc = _validation(0.009, [0.001, 0.001, 0.001, 0.001])
    broad = _validation(0.019, [0.049, 0.049, 0.049, 0.049])
    assert broad_training_admitted(bc, broad)
    broad["mse"][1] = 0.051
    assert not broad_training_admitted(bc, broad)
    broad["mse"][1] = 0.049
    bc["weighted_mse"] = 0.011
    assert not broad_training_admitted(bc, broad)
