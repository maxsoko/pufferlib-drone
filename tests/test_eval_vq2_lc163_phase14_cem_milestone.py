from __future__ import annotations

import scripts.eval_vq2_lc163_phase14_cem_milestone as lc163


def test_lc163_source_lock() -> None:
    parent = lc163.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc163.TARGET_RAW_INDEX == 15


def test_lc163_complete_gain_contract() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    complete = {
        "target_passes": 128, "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert lc163.base.choose_candidate([baseline, complete]) is complete
