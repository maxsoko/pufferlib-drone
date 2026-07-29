from __future__ import annotations

import numpy as np

from scripts.train_vq2_measured_two_gate_dagger_second import (
    CONFIG,
    ThreeSourcePublicPhaseDataset,
)


def test_three_source_dataset_interleaves_all_validation_sources() -> None:
    dataset = ThreeSourcePublicPhaseDataset()
    assert dataset.agents == 192
    assert np.array_equal(dataset.source_for_agent[:9], np.tile([0, 1, 2], 3))
    assert np.array_equal(dataset.local_agent[:9], np.repeat([0, 1, 2], 3))
    assert np.array_equal(dataset.source_for_agent[-18:], np.tile([0, 1, 2], 6))
    assert CONFIG.validation_agents == 18
