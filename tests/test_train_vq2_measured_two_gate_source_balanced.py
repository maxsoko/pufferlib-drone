from __future__ import annotations

import numpy as np

from scripts.train_vq2_measured_two_gate_source_balanced import (
    CONFIG,
    BalancedPublicPhaseDataset,
    sf058_audit_passes,
)


def test_balanced_dataset_repeats_short_phase1_source_fivefold() -> None:
    dataset = BalancedPublicPhaseDataset()
    assert dataset.agents == 512
    assert len(dataset.sources) == 8
    assert len({id(source) for source in dataset.sources[-5:]}) == 1
    assert np.array_equal(dataset.source_for_agent[-64:], np.tile(range(8), 8))
    assert CONFIG.validation_agents == 64
    assert CONFIG.learning_rate == 1e-5


def test_sf058_audit_has_strict_source_specific_gate() -> None:
    good = {
        name: {"weighted_mse": 0.01, "mse": [0.05, 0.05, 0.05, 0.05]}
        for name in ("phase_zero", "gate2")
    }
    assert sf058_audit_passes(good)
    bad = {**good, "gate2": {**good["gate2"], "weighted_mse": 0.0101}}
    assert not sf058_audit_passes(bad)
