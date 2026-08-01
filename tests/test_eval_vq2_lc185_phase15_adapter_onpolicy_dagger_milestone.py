from __future__ import annotations

import torch

import scripts.eval_vq2_lc185_phase15_adapter_onpolicy_dagger_milestone as lc185


def test_lc185_source_lock() -> None:
    parent = lc185.verify_inputs()
    assert parent["numerically_admitted"]
    assert lc185.candidate_payload()["numerically_admitted"]


def test_lc185_only_adapter_changed() -> None:
    parent = lc185.parent_payload()["model_state"]
    candidate = lc185.candidate_payload()["model_state"]
    prefixes = ("phase_adapter_cell.", "phase_adapter_output.")
    assert all(torch.equal(value, candidate[name]) for name, value in parent.items() if not name.startswith(prefixes))
    assert any(not torch.equal(value, candidate[name]) for name, value in parent.items() if name.startswith(prefixes))


def test_lc185_selection_is_strict() -> None:
    baseline = {"target_passes": 0, "pre_target_terminals": 128}
    candidate = {
        "target_passes": 128,
        "paired_target_gains_vs_baseline": 128,
        "paired_target_losses_vs_baseline": 0,
        "transport_pass": True,
        "pre_target_terminals": 0,
    }
    assert lc185.choose_candidate([baseline, candidate]) is candidate
    candidate["target_passes"] = 127
    assert lc185.choose_candidate([baseline, candidate]) is None
