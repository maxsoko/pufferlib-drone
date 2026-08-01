from __future__ import annotations

import torch

import scripts.eval_vq2_lc126_phase6_onpolicy_coordinate_es as lc126


def test_lc126_source_lock_and_coordinate_contract() -> None:
    parent = lc126.verify_inputs()["model_state"]
    assert len(lc126.BIASES) == 8
    assert lc126.GROUP_SIZE * len(lc126.BIASES) == 256
    baseline = lc126.candidate_state_for_index(parent, 0)
    assert all(torch.equal(baseline[name], parent[name]) for name in parent)
    for index in range(1, len(lc126.BIASES)):
        candidate = lc126.candidate_state_for_index(parent, index)
        changed = [
            name for name in parent
            if not torch.equal(candidate[name], parent[name])
        ]
        assert changed == ["indexed_phase_residual_output_bias"]
        difference = (
            candidate["indexed_phase_residual_output_bias"]
            - parent["indexed_phase_residual_output_bias"]
        )
        assert torch.count_nonzero(difference[:6]) == 0
        assert torch.count_nonzero(difference[7:]) == 0
        assert torch.allclose(
            difference[6], torch.tensor(lc126.BIASES[index][1])
        )
