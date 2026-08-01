from __future__ import annotations

import numpy as np

import scripts.collect_vq2_lc125_lc123_phase8_9_rescue_features as lc125


def test_lc125_source_lock_and_feature_contract() -> None:
    payload = lc125.verify_inputs()
    assert payload["numerically_admitted"]
    assert lc125.FEATURE_DTYPE.itemsize > 0
    assert lc125.PHASE_MIN == 8
    assert lc125.PHASE_MAX_EXCLUSIVE == 10


def test_lc125_intervention_mask_is_group_and_phase_exact() -> None:
    active = np.ones(lc125.TOTAL_AGENTS, dtype=bool)
    phase = np.full(lc125.TOTAL_AGENTS, 8, dtype=np.int32)
    phase[lc125.GROUP_SIZE + 1] = 7
    phase[lc125.GROUP_SIZE + 2] = 10
    active[lc125.GROUP_SIZE + 3] = False
    mask = lc125.intervention_mask(active, phase)
    assert not mask[: lc125.GROUP_SIZE].any()
    assert not mask[lc125.GROUP_SIZE + 1]
    assert not mask[lc125.GROUP_SIZE + 2]
    assert not mask[lc125.GROUP_SIZE + 3]
    assert int(mask.sum()) == lc125.GROUP_SIZE - 3
