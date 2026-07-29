from __future__ import annotations

import numpy as np

from scripts.train_vq2_measured_two_gate_dagger_phase1 import (
    CONFIG,
    FourSourcePublicPhaseDataset,
)


def test_four_source_dataset_and_phase_balance_are_fixed() -> None:
    dataset = FourSourcePublicPhaseDataset()
    assert dataset.agents == 256
    assert np.array_equal(dataset.source_for_agent[:8], np.tile([0, 1, 2, 3], 2))
    assert np.array_equal(dataset.source_for_agent[-32:], np.tile([0, 1, 2, 3], 8))
    assert CONFIG.validation_agents == 32
    assert CONFIG.phase_zero_loss_weight == 1.0
    assert CONFIG.gate2_loss_weight == 2.0
