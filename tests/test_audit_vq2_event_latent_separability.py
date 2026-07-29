import numpy as np

from scripts.audit_vq2_event_latent_separability import (
    _average_precision,
    _fit_ridge_probe,
    _roc_auc,
    _stratified_group_split,
    _validation_threshold,
)


def test_rank_metrics_are_exact_for_separated_scores():
    labels = np.asarray([0, 1, 0, 1], dtype=bool)
    scores = np.asarray([-2.0, 2.0, -1.0, 1.0])
    assert _roc_auc(labels, scores) == 1.0
    assert _average_precision(labels, scores) == 1.0


def test_roc_auc_uses_half_credit_for_ties():
    labels = np.asarray([0, 1], dtype=bool)
    scores = np.asarray([0.0, 0.0])
    assert _roc_auc(labels, scores) == 0.5


def test_group_split_is_disjoint_deterministic_and_phase_covered():
    groups = np.repeat(np.arange(60), 2)
    phases = np.tile(np.arange(1, 7), 20)
    first = _stratified_group_split(
        groups,
        phases,
        seed=628,
        validation_fraction=0.125,
        test_fraction=0.25,
    )
    second = _stratified_group_split(
        groups,
        phases,
        seed=628,
        validation_fraction=0.125,
        test_fraction=0.25,
    )
    assert np.array_equal(first, second)
    for group in np.unique(groups):
        assert np.unique(first[groups == group]).size == 1
    for phase in range(1, 7):
        for split in range(3):
            assert np.any((phases == phase) & (first == split))


def test_closed_form_probe_selects_a_separable_signal():
    rng = np.random.default_rng(628)
    groups = np.repeat(np.arange(60), 5)
    labels = np.tile(np.asarray([0, 0, 0, 0, 1], dtype=bool), 60)
    event_split = np.repeat(np.asarray([0] * 36 + [1] * 12 + [2] * 12), 5)
    signal = (labels.astype(np.float64) * 2.0 - 1.0)[:, None]
    features = np.concatenate((signal + rng.normal(0.0, 0.05, signal.shape), rng.normal(size=(300, 3))), 1)
    scores, summary = _fit_ridge_probe(
        features,
        labels,
        event_split,
        ridge_values=(0.01, 0.1, 1.0),
    )
    assert summary["selected_ridge"] in (0.01, 0.1, 1.0)
    assert _roc_auc(labels[event_split == 2], scores[event_split == 2]) == 1.0
    threshold = _validation_threshold(labels[event_split == 1], scores[event_split == 1])
    assert np.isfinite(threshold)
    assert np.unique(event_split[np.isin(groups, groups[event_split == 2])]).tolist() == [2]
