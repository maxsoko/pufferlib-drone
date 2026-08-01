from __future__ import annotations

import scripts.eval_vq2_lc161_phase13_cem_milestone as lc161


def test_lc161_source_lock() -> None:
    parent = lc161.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc161.TARGET_RAW_INDEX == 14


def test_lc161_complete_gain_contract() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    complete = {
        "target_passes": 128, "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert lc161.base.choose_candidate([baseline, complete]) is complete
