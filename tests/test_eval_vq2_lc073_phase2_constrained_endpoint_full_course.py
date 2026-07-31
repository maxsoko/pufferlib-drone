from __future__ import annotations

import torch

import scripts.eval_vq2_lc073_phase2_constrained_endpoint_full_course as lc073


def test_selected_state_changes_only_phase2_decoder() -> None:
    parent = lc073.verify_inputs()
    state = parent["model_state"]
    candidate = lc073.build_selected_candidate_state(state)
    changed = []
    for name, value in state.items():
        if not torch.equal(value, candidate[name]):
            changed.append(name)
    assert changed == [
        "indexed_phase_residual_output",
        "indexed_phase_residual_output_bias",
    ]
    for name in changed:
        mask = torch.ones(candidate[name].shape[0], dtype=torch.bool)
        mask[lc073.base.TARGET_PHASE] = False
        assert torch.equal(candidate[name][mask], state[name][mask])


def test_selected_decoder_matches_lc070_exactly() -> None:
    parent = lc073.verify_inputs()
    candidate = lc073.build_selected_candidate_state(parent["model_state"])
    endpoint = lc073.endpoint_decoder()
    assert torch.equal(
        candidate["indexed_phase_residual_output"][lc073.base.TARGET_PHASE],
        endpoint["output_weight"],
    )
    assert torch.equal(
        candidate["indexed_phase_residual_output_bias"][lc073.base.TARGET_PHASE],
        endpoint["output_bias"],
    )

