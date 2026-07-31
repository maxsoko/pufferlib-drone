from __future__ import annotations

import torch

import scripts.eval_vq2_lc058_phase2_bias_milestone as lc058


def test_bias_matrix_repeats_each_candidate_group() -> None:
    matrix = lc058.bias_matrix()
    assert matrix.shape == (lc058.TOTAL_AGENTS, 4)
    for group, (_, bias) in enumerate(lc058.BIAS_CANDIDATES):
        expected = torch.tensor(bias, dtype=torch.float32).expand(lc058.GROUP_SIZE, -1)
        assert torch.equal(matrix[lc058.group_slice(group)], expected)


def test_phase2_bias_is_exact_pre_tanh_surgery() -> None:
    pre = torch.tensor([[0.2, -0.3, 0.1, 0.0], [0.2, -0.3, 0.1, 0.0]])
    held = torch.tensor([[2.0 / lc058.OFFICIAL_PROGRESS_SCALE], [1.0 / lc058.OFFICIAL_PROGRESS_SCALE]])
    delta = torch.tensor([[-0.01, 0.02, 0.0, 0.0], [-0.01, 0.02, 0.0, 0.0]])
    actual = lc058.apply_phase2_bias(pre, held, delta)
    assert torch.equal(actual[0], torch.tanh(pre[0] + delta[0]))
    assert torch.equal(actual[1], torch.tanh(pre[1]))


def test_selection_requires_gain_without_terminal_regression() -> None:
    baseline = {
        "candidate_index": 0, "gate3_passes": 18,
        "pre_gate3_terminals": 14, "bias_l2": 0.0,
        "transport_pass": True,
    }
    worse_safety = {
        "candidate_index": 1, "gate3_passes": 21,
        "pre_gate3_terminals": 15, "bias_l2": 0.0025,
        "transport_pass": True,
    }
    smaller = {
        "candidate_index": 2, "gate3_passes": 20,
        "pre_gate3_terminals": 12, "bias_l2": 0.005,
        "transport_pass": True,
    }
    larger = {
        "candidate_index": 3, "gate3_passes": 20,
        "pre_gate3_terminals": 12, "bias_l2": 0.01,
        "transport_pass": True,
    }
    assert lc058.choose_candidate([baseline, worse_safety, larger, smaller]) is smaller
    assert lc058.choose_candidate([baseline, worse_safety]) is None
