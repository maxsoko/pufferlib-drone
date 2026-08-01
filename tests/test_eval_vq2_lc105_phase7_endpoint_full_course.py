from __future__ import annotations

import torch

import scripts.eval_vq2_lc105_phase7_endpoint_full_course as lc105


def test_lc105_source_lock_and_exact_candidate_state() -> None:
    parent = lc105.verify_inputs()
    candidate = lc105.candidate_state(parent["model_state"])
    fitted = lc105.endpoint.endpoint_decoder()
    assert torch.equal(
        candidate["indexed_phase_residual_output"][7], fitted["output_weight"]
    )
    assert torch.equal(
        candidate["indexed_phase_residual_output_bias"][7], fitted["output_bias"]
    )
    for phase in range(24):
        if phase == 7:
            continue
        assert torch.equal(
            candidate["indexed_phase_residual_output"][phase],
            parent["model_state"]["indexed_phase_residual_output"][phase],
        )


def test_lc105_promotion_requires_progress_and_safety() -> None:
    parent = {
        "transport_pass": True, "mean_gates_passed": 3.0,
        "promotion_target_passes": 1, "crash_rate": 0.1,
        "maximum_raw_index": 9,
    }
    candidate = {
        "transport_pass": True,
        "mean_gates_passed": 3.0 + lc105.MINIMUM_MEAN_GATE_GAIN,
        "promotion_target_passes": 2, "crash_rate": 0.1,
        "maximum_raw_index": 9,
    }
    assert lc105.choose_candidate(parent, candidate)
    candidate["crash_rate"] = 0.11
    assert not lc105.choose_candidate(parent, candidate)
