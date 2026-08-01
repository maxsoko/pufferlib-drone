from __future__ import annotations

import scripts.eval_vq2_lc190_phase16_adapter_cem_milestone as lc190


def test_lc190_source_lock() -> None:
    assert lc190.verify_inputs()["numerically_admitted"]
    assert lc190.candidate_payload()["frozen_non_phase16_state_exact"]


def test_lc190_selection_is_strict() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    candidate = {
        "target_passes": 128,
        "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0,
        "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert lc190.choose_candidate([baseline, candidate]) is candidate
    candidate["target_passes"] = 127
    assert lc190.choose_candidate([baseline, candidate]) is None
