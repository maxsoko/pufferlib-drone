from __future__ import annotations

import torch

import scripts.eval_vq2_lc084_phase6_success_bias_full_course as lc084


def test_lc084_changes_only_phase6_decoder_bias() -> None:
    parent = lc084.verify_inputs()
    state = parent["model_state"]
    candidate = lc084.build_selected_candidate_state(state)
    changed = [name for name in state if not torch.equal(state[name], candidate[name])]
    assert changed == ["indexed_phase_residual_output_bias"]
    delta = candidate[changed[0]] - state[changed[0]]
    mask = torch.ones(delta.shape[0], dtype=torch.bool)
    mask[lc084.TARGET_PHASE] = False
    assert torch.equal(delta[mask], torch.zeros_like(delta[mask]))
    assert torch.allclose(
        delta[lc084.TARGET_PHASE], torch.tensor(lc084.BIAS), atol=1e-8, rtol=0.0
    )


def test_lc084_promotion_contract_targets_raw_index7() -> None:
    assert lc084.base.GROUP_SIZE == 64
    assert lc084.base.NUM_GATES == 24
    assert lc084.TARGET_RAW_INDEX == 7
    assert lc084.MINIMUM_TARGET_PASS_GAIN == 1
    assert lc084.MINIMUM_MEAN_GATE_GAIN == 1.0 / 64.0


def test_lc084_choose_candidate_requires_downstream_gain() -> None:
    parent = {
        "transport_pass": True, "mean_gates_passed": 3.0,
        "promotion_target_passes": 1, "crash_rate": 0.2,
        "maximum_raw_index": 7,
    }
    candidate = {
        "transport_pass": True, "mean_gates_passed": 3.0 + 1.0 / 64.0,
        "promotion_target_passes": 2, "crash_rate": 0.2,
        "maximum_raw_index": 7,
    }
    assert lc084.choose_candidate(parent, candidate)
    candidate["promotion_target_passes"] = 1
    assert not lc084.choose_candidate(parent, candidate)
