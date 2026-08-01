from __future__ import annotations

import torch

import scripts.eval_vq2_lc097_late_phase_endpoint_local_bracket as lc097


def test_lc097_local_bracket_contract() -> None:
    assert lc097.ALPHAS == (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
    assert lc097.GROUP_SIZE * lc097.GROUPS == 448
    assert lc097.ADMISSIBLE_PHASES == (6, 7, *range(10, 24))
    assert 8 not in lc097.ADMISSIBLE_PHASES and 9 not in lc097.ADMISSIBLE_PHASES


def test_lc097_interpolates_only_admissible_output_rows() -> None:
    parent, endpoint = lc097.verify_inputs()
    state = lc097.candidate_state(parent["model_state"], endpoint["model_state"], 0.1)
    for name in lc097.INTERPOLATED_NAMES:
        for phase in range(33):
            if phase in lc097.ADMISSIBLE_PHASES:
                expected = torch.lerp(
                    parent["model_state"][name][phase],
                    endpoint["model_state"][name][phase],
                    0.1,
                )
                assert torch.allclose(state[name][phase], expected)
            else:
                assert torch.equal(state[name][phase], parent["model_state"][name][phase])


def test_lc097_selection_requires_progress_and_no_crash_regression() -> None:
    baseline = {
        "alpha": 0.0, "transport_pass": True, "mean_gate_advance": 0.1,
        "one_gate_passes": 5, "two_gate_passes": 1, "crash_rate": 0.1,
    }
    candidate = {
        "alpha": 0.01, "transport_pass": True, "mean_gate_advance": 0.2,
        "one_gate_passes": 7, "two_gate_passes": 2, "crash_rate": 0.1,
    }
    assert lc097.choose_candidate([baseline, candidate]) is candidate
    candidate["crash_rate"] = 0.2
    assert lc097.choose_candidate([baseline, candidate]) is None
