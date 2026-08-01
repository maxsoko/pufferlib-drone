from __future__ import annotations

import numpy as np

import scripts.train_vq2_lc096_late_phase_multihead_endpoint as lc096


def test_lc096_all_late_heads_fit_once() -> None:
    assert lc096.PHASES == tuple(range(6, 24))
    assert len(lc096.PHASES) == 18
    assert lc096.MINIMUM_PHASE_IMPROVEMENT == 1.25
    assert lc096.MAXIMUM_PHASE_DELTA_L2 == 64.0


def test_lc096_agent_split_and_weights() -> None:
    present = np.arange(10)
    training, validation = lc096.split_agents(present)
    assert np.array_equal(validation, np.array([0, 5]))
    assert not np.intersect1d(training, validation).size
    rows = np.array([1, 1, 2, 2, 2, 3])
    weights = lc096.equal_agent_weights(rows, np.array([1, 2]))
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[rows == 1].sum(), 0.5)
    assert np.isclose(weights[rows == 2].sum(), 0.5)


def test_lc096_dataset_report_is_source_locked_without_large_feature_read() -> None:
    import json
    from scripts.eval_vq2_variable_gate_oracle import sha256_path

    assert sha256_path(lc096.DATASET_REPORT) == lc096.DATASET_REPORT_SHA256
    report = json.loads(lc096.DATASET_REPORT.read_text())
    assert report["training_dataset_admitted"]
    assert all(report["feature_phase_records"][phase] >= 2_000 for phase in lc096.PHASES)
