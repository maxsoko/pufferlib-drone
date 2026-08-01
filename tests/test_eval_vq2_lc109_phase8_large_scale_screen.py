from __future__ import annotations

import scripts.eval_vq2_lc109_phase8_large_scale_screen as lc109


def test_lc109_source_lock_and_large_scale_contract() -> None:
    lc109.configure()
    lc109.verify_inputs()
    assert lc109.ALPHAS == (0.0, 1.5, 2.0, 3.0)
    assert lc109.base.TARGET_PHASE == 8
    assert lc109.base.TARGET_RAW_INDEX == 9
    assert lc109.base.pairwise.PAIR_SIZE == 256
    assert lc109.milestone.TOTAL_AGENTS == 512
