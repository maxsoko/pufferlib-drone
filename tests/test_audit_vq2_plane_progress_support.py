import numpy as np

from scripts.audit_vq2_plane_progress_support import (
    _correlation,
    _direct_score_result,
)


def test_correlation_handles_exact_and_constant_inputs():
    values = np.asarray([0.0, 1.0, 2.0])
    assert _correlation(values, values) == 1.0
    assert _correlation(values.reshape(1, -1), values.reshape(1, -1)) == 1.0
    assert _correlation(values, np.zeros(3)) == 0.0


def test_direct_score_reports_within_window_ranking():
    labels = np.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=bool)
    scores = np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    split = np.asarray([0, 1, 2], dtype=np.int8)
    result = _direct_score_result(labels, scores, split)
    assert result["test"]["roc_auc"] == 1.0
    assert result["test_within_window_all_negative_rank_fraction"] == 1.0
