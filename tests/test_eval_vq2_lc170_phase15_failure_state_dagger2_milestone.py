from __future__ import annotations

import scripts.eval_vq2_lc170_phase15_failure_state_dagger2_milestone as lc170


def test_lc170_source_lock() -> None:
    parent = lc170.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc170.previous.TARGET_RAW_INDEX == 16


def test_lc170_complete_gain_contract() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    complete = {
        "target_passes": 128, "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert lc170.previous.previous.prior.choose_candidate([baseline, complete]) is complete
