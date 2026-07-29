from __future__ import annotations

import numpy as np
import torch

from scripts.train_vq2_measured_two_gate_dagger import (
    CONFIG,
    ORACLE_HORIZON,
    TwoSourcePublicPhaseDataset,
)


def test_two_source_dataset_interleaves_real_episode_starts() -> None:
    dataset = TwoSourcePublicPhaseDataset()
    assert dataset.agents == 128
    assert dataset.time_steps == ORACLE_HORIZON == 1472
    assert np.array_equal(dataset.source_for_agent[:6], [0, 1, 0, 1, 0, 1])
    assert np.array_equal(dataset.local_agent[:6], [0, 0, 1, 1, 2, 2])
    assert np.array_equal(dataset.source_for_agent[-16:], np.tile([0, 1], 8))
    assert CONFIG.validation_agents == 16
    observation, action, valid, gate2 = dataset.chunk(
        np.asarray([0, 1]), 480, 512, device=torch.device("cpu")
    )
    assert observation.shape == (2, 32, 4119)
    assert action.shape == (2, 32, 4)
    assert valid.shape == gate2.shape == (2, 32)
    assert bool(valid[0].all())
    dagger_valid_steps = max(int(dataset.lengths[1]) - 480, 0)
    assert not bool(valid[1, dagger_valid_steps:].any())
