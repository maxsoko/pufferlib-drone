from __future__ import annotations

import numpy as np

import scripts.train_vq2_vg064_indexed_intervention_ridge as core
import scripts.train_vq2_vg065_group_stratified_intervention_ridge as vg065


def test_group_stratified_holdout_covers_every_native_count_residue() -> None:
    agents = np.arange(512, dtype=np.uint16)
    held = agents[core.validation_mask(agents)]
    assert held.size == 64
    assert np.bincount(held % 8, minlength=8).tolist() == [8] * 8
    # Phase 11 exists only in the native count-12 residue (agent % 8 == 1).
    assert int((held % 8 == 1).sum()) == 8


def test_vg065_changes_identity_not_fit_math() -> None:
    assert vg065.SEED > core.SEED
    assert core.VALIDATION_AGENT_GROUP_SIZE == 8
    assert core.RIDGES == (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    assert core.TARGET_PHASES == tuple(range(1, 12))
