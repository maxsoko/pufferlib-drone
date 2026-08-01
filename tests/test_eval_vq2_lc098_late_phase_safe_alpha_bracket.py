from __future__ import annotations

import scripts.eval_vq2_lc098_late_phase_safe_alpha_bracket as lc098


def test_lc098_refined_safe_bracket_contract() -> None:
    assert lc098.ALPHAS == (0.0, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
    assert lc098.GROUP_SIZE == 96
    assert lc098.GROUP_SIZE * len(lc098.ALPHAS) == 672
    assert lc098.MINIMUM_MEAN_ADVANCE_GAIN == 0.03
    assert lc098.MINIMUM_ONE_GATE_PASS_GAIN == 2


def test_lc098_source_locks_safe_and_unsafe_lc097_points() -> None:
    lc098.verify_inputs()
