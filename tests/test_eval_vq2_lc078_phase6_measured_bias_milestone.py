from __future__ import annotations

import torch

import scripts.eval_vq2_lc078_phase6_measured_bias_milestone as lc078


def test_bias_surgery_changes_only_phase6_output_bias() -> None:
    parent = lc078.verify_inputs()["model_state"]
    candidate = lc078.candidate_state_for_index(parent, 2)
    changed = [name for name in parent if not torch.equal(parent[name], candidate[name])]
    assert changed == ["indexed_phase_residual_output_bias"]
    keep = torch.arange(candidate[changed[0]].shape[0]) != lc078.TARGET_PHASE
    assert torch.equal(candidate[changed[0]][keep], parent[changed[0]][keep])


def test_lc078_is_one_index7_vector() -> None:
    assert lc078.TARGET_PHASE == 6
    assert lc078.TARGET_RAW_INDEX == 7
    assert lc078.GROUP_SIZE * len(lc078.BIASES) == 256
    assert any(values[1] > 0 for _, values in lc078.BIASES)
    assert any(values[1] < 0 for _, values in lc078.BIASES)

