from __future__ import annotations

import scripts.eval_vq2_lc196_phase16_17_joint_interpolation_bracket as target


def test_grid_contract() -> None:
    assert len(target.SCALE_PAIRS) == 13
    assert target.SCALE_PAIRS[0] == (0.0, 0.0)
    assert target.SCALE_PAIRS[3] == (0.0, 1.0)
    assert target.GROUP_SIZE * len(target.SCALE_PAIRS) == 416
    assert target.LC195_REPORT_SHA256 == "b046188466d304ed900853499ae9ba2084370870c5bd4902313c1ee716b66a5b"


def test_selection_is_strict_and_prefers_phase16_preservation() -> None:
    baseline = {
        "target_passes": 0, "pre_target_terminals": 32,
        "maximum_raw_index_distribution": {"17": 32},
    }
    first = {
        "target_passes": 32, "paired_target_gains_vs_baseline": 32,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 0, "phase16_fit_scale": 0.0025,
        "phase17_fit_scale": 1.0,
    }
    second = {**first, "phase16_fit_scale": 0.0, "phase17_fit_scale": 1.0}
    assert target.choose_candidate([baseline, first, second]) is second
    second["target_passes"] = 31
    assert target.choose_candidate([baseline, first, second]) is first
