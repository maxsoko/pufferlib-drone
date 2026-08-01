from __future__ import annotations

import scripts.eval_vq2_lc188_phase15_adapter_onpolicy_dagger2_milestone as lc188


def test_lc188_source_lock() -> None:
    assert lc188.verify_inputs()["numerically_admitted"]
    assert lc188.candidate_payload()["numerically_admitted"]


def test_lc188_selection_is_strict() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    candidate = {
        "target_passes": 128,
        "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0,
        "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert lc188.choose_candidate([baseline, candidate]) is candidate
    candidate["target_passes"] = 127
    assert lc188.choose_candidate([baseline, candidate]) is None
