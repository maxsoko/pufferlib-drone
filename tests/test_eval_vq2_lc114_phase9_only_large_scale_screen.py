from __future__ import annotations

import scripts.eval_vq2_lc114_phase9_only_large_scale_screen as lc114


def test_lc114_source_lock_and_final_scale_contract() -> None:
    lc114.configure()
    lc114.verify_inputs()
    assert lc114.ALPHAS == (0.0, 0.30, 0.50, 1.0)
    assert lc114.base.TARGET_PHASE == 9
    assert lc114.base.base.TARGET_RAW_INDEX == 10
    assert lc114.milestone.TOTAL_AGENTS == 512
