from __future__ import annotations

import torch

import scripts.eval_vq2_lc174_phase15_failure_state_dagger3_milestone as lc174


def test_lc174_source_lock_and_isolation() -> None:
    parent = lc174.verify_inputs()
    candidate = torch.load(lc174.CANDIDATE_CHECKPOINT, map_location="cpu", weights_only=False)
    assert parent["numerically_admitted"]
    assert candidate["numerically_admitted"]


def test_lc174_pair_contract() -> None:
    assert len(lc174.BIASES) == 2
    assert lc174.BIASES[0][1] == (0.0,) * 4
    assert lc174.BIASES[1][1] == (0.0,) * 4
