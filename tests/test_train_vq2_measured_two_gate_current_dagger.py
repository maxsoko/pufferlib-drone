from __future__ import annotations

import numpy as np

from scripts.train_vq2_measured_two_gate_current_dagger import (
    CONFIG,
    CURRENT_REPETITIONS,
    CurrentDaggerDataset,
    source_audits_pass,
)


def test_current_dataset_balances_gate2_without_splicing_histories() -> None:
    dataset = CurrentDaggerDataset()
    assert dataset.agents == 448
    assert len(dataset.sources) == 7
    assert len({id(source) for source in dataset.sources[1:]}) == 1
    assert CURRENT_REPETITIONS == 6
    assert np.array_equal(dataset.source_for_agent[-56:], np.tile(range(7), 8))
    assert CONFIG.validation_agents == 56
    assert CONFIG.learning_rate == 2e-5


def test_source_audits_require_both_clean_and_current_sources() -> None:
    good_partition = {
        "weighted_mse": 0.01,
        "mse": [0.05, 0.05, 0.05, 0.05],
    }
    good = {
        source: {partition: dict(good_partition) for partition in ("phase_zero", "gate2")}
        for source in ("sf049_clean", "sf062_current")
    }
    assert source_audits_pass(good)
    bad = {
        **good,
        "sf062_current": {
            **good["sf062_current"],
            "gate2": {**good_partition, "weighted_mse": 0.0101},
        },
    }
    assert not source_audits_pass(bad)
