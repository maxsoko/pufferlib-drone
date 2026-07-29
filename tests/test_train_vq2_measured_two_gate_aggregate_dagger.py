from __future__ import annotations

import numpy as np

from scripts.train_vq2_measured_two_gate_aggregate_dagger import (
    CONFIG,
    REPETITIONS,
    AggregateDaggerDataset,
    source_audits_pass,
)


def test_aggregate_dataset_balances_two_failure_distributions() -> None:
    dataset = AggregateDaggerDataset()
    assert dataset.agents == 448
    assert len(dataset.sources) == 7
    assert len({id(source) for source in dataset.sources[1:4]}) == 1
    assert len({id(source) for source in dataset.sources[4:]}) == 1
    assert REPETITIONS == 3
    assert np.array_equal(dataset.source_for_agent[-56:], np.tile(range(7), 8))
    assert CONFIG.validation_agents == 56
    assert CONFIG.learning_rate == 2e-5


def test_source_audits_require_clean_prior_and_underturn_sources() -> None:
    good_partition = {"weighted_mse": 0.01, "mse": [0.05] * 4}
    sources = ("sf049_clean", "sf062_prior_crash", "sf065_underturn")
    good = {
        source: {partition: dict(good_partition) for partition in ("phase_zero", "gate2")}
        for source in sources
    }
    assert source_audits_pass(good)
    bad = {
        **good,
        "sf065_underturn": {
            **good["sf065_underturn"],
            "gate2": {**good_partition, "mse": [0.0501, 0.0, 0.0, 0.0]},
        },
    }
    assert not source_audits_pass(bad)
