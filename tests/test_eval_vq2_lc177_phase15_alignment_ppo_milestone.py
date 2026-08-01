from __future__ import annotations

import torch

import scripts.eval_vq2_lc177_phase15_alignment_ppo_milestone as lc177


def test_lc177_source_lock_and_isolation() -> None:
    parent = lc177.verify_inputs()
    candidate = torch.load(lc177.CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    assert parent["numerically_admitted"]
    assert candidate["onpolicy_rollout_iteration"] == 3


def test_lc177_pair_contract() -> None:
    assert len(lc177.BIASES) == 2
    assert lc177.BIASES[0][1] == (0.0,) * 4
    assert lc177.BIASES[1][1] == (0.0,) * 4
    assert lc177.SELECTED_PARAMETER_DELTA_L2 > 0.0
