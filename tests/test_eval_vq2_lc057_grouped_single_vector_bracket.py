from __future__ import annotations

import pytest

import scripts.eval_vq2_lc057_grouped_single_vector_bracket as lc057


def test_group_layout_covers_four_disjoint_paired_blocks() -> None:
    assert lc057.TOTAL_AGENTS == 128
    assert lc057.THREADS == 128
    assert [lc057.group_slice(i) for i in range(lc057.GROUPS)] == [
        slice(0, 32), slice(32, 64), slice(64, 96), slice(96, 128)
    ]


def test_group_slice_rejects_out_of_range_candidate() -> None:
    with pytest.raises(ValueError, match="outside"):
        lc057.group_slice(-1)
    with pytest.raises(ValueError, match="outside"):
        lc057.group_slice(lc057.GROUPS)


def test_speed_gate_targets_material_cycle_reduction() -> None:
    assert lc057.MINIMUM_SPEEDUP == 2.0
    assert lc057.GROUP_SIZE == 32
