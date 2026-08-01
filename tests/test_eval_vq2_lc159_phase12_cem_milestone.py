from __future__ import annotations
import scripts.eval_vq2_lc159_phase12_cem_milestone as lc159


def test_lc159_source_lock() -> None:
    parent = lc159.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc159.TARGET_RAW_INDEX == 13


def test_lc159_complete_gain_contract() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    complete = {
        "target_passes": 128, "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0, "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert base_choose([baseline, complete]) is complete


def base_choose(items):
    return lc159.base.choose_candidate(items)
