from __future__ import annotations

import scripts.eval_vq2_lc072_phase2_constrained_endpoint_confirmation as lc072


def test_lc072_is_a_fresh_paired_128_seed_confirmation() -> None:
    assert lc072.GROUP_SIZE == 128
    assert lc072.ALPHAS == (0.0, 1.0)
    assert lc072.GROUP_SIZE * len(lc072.ALPHAS) == 256
    assert lc072.SEED != lc072.base.SEED


def test_lc071_authority_is_source_locked() -> None:
    report = lc072.verify_inputs()
    assert report["model_state"]

