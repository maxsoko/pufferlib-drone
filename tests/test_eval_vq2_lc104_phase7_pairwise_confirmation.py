from __future__ import annotations

import scripts.eval_vq2_lc104_phase7_pairwise_confirmation as lc104


def test_lc104_source_lock_and_contract() -> None:
    lc104.configure()
    lc104.verify_inputs()
    assert lc104.GROUP_SIZE == 128
    assert lc104.ALPHAS == (0.0, 1.0)
    assert lc104.SEED == 432_040
    assert lc104.MINIMUM_PASS_GAIN == 1
    assert lc104.prior.PAIR_SIZE == 256
    assert lc104.milestone.TOTAL_AGENTS == 256
