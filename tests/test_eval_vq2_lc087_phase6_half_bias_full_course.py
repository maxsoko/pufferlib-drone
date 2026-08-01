from __future__ import annotations

import torch

import scripts.eval_vq2_lc087_phase6_half_bias_full_course as lc087


def test_lc087_higher_power_full_course_contract() -> None:
    assert lc087.GROUP_SIZE == 128
    assert lc087.TARGET_RAW_INDEX == 7
    assert lc087.MINIMUM_TARGET_PASS_GAIN == 1
    assert lc087.MINIMUM_MEAN_GATE_GAIN == 1.0 / 128.0


def test_lc087_source_lock_and_half_bias_state() -> None:
    parent = lc087.verify_inputs()
    state = parent["model_state"]
    candidate = lc087.build_selected_candidate_state(state)
    delta = (
        candidate["indexed_phase_residual_output_bias"][lc087.TARGET_PHASE]
        - state["indexed_phase_residual_output_bias"][lc087.TARGET_PHASE]
    )
    assert torch.allclose(delta, torch.tensor(lc087.BIAS), atol=1e-8, rtol=0.0)


def test_lc087_choose_candidate_requires_index7_gain() -> None:
    parent = {
        "transport_pass": True, "mean_gates_passed": 3.0,
        "promotion_target_passes": 1, "crash_rate": 0.2,
        "maximum_raw_index": 7,
    }
    candidate = {
        "transport_pass": True, "mean_gates_passed": 3.0 + 1.0 / 128.0,
        "promotion_target_passes": 2, "crash_rate": 0.2,
        "maximum_raw_index": 7,
    }
    assert lc087.choose_candidate(parent, candidate)
    candidate["promotion_target_passes"] = 1
    assert not lc087.choose_candidate(parent, candidate)
