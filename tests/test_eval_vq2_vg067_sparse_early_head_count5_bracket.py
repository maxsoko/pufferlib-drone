from __future__ import annotations

import scripts.eval_vq2_vg066_sparse_early_head_count5_bracket as vg066
import scripts.eval_vq2_vg067_sparse_early_head_count5_bracket_repaired as vg067


def test_vg067_retains_sparse_math_with_fresh_fixture() -> None:
    assert vg067.OFFSETS == (200,)
    assert vg067.SEEDS == (429202,)
    assert vg066.SPARSE_PHASES == (1, 2, 3)
    assert vg066.ALPHAS == (0.0, 0.25, 0.50, 0.75, 1.0)
    assert vg066.MAX_STEPS == 360 * 64


def test_vg067_adds_failure_and_wrapper_to_source_surface() -> None:
    vg067.configure()
    assert vg067.FAILURE in vg066.EXTRA_SOURCE_PATHS
    assert vg067.TEST in vg066.EXTRA_SOURCE_PATHS
