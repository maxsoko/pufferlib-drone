from __future__ import annotations

import scripts.collect_vq2_lc074_phase6_capped_student_features as lc074


def test_lc074_uses_the_reduced_cycle_contract() -> None:
    assert lc074.AGENTS == 256
    assert lc074.THREADS == 32
    assert lc074.TARGET_PHASE == 6
    assert lc074.RECORD_STOP == 20_000
    assert lc074.STEP_LIMIT == 6_000


def test_lc073_parent_is_source_locked() -> None:
    lc074.verify_inputs()

