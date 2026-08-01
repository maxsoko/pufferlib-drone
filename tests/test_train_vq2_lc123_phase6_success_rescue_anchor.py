from __future__ import annotations

import numpy as np
import torch

import scripts.train_vq2_lc123_phase6_success_rescue_anchor as lc123


def test_lc123_source_lock_and_split_contract() -> None:
    parent = lc123.verify_inputs()
    assert parent["numerically_admitted"]
    present = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    outcome = np.zeros(7, dtype=np.uint8)
    outcome[[4, 5, 6]] = 1
    training, validation = lc123.stratified_agent_split(present, outcome)
    assert set(training).isdisjoint(set(validation))
    assert set(training) | set(validation) == set(present)
    assert set(outcome[training]) == {0, 1}
    assert set(outcome[validation]) == {0, 1}


def test_lc123_anchors_failures_and_labels_successes() -> None:
    teacher = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    parent = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    target = lc123.anchored_target(
        teacher, parent, torch.tensor([True, False])
    )
    assert torch.equal(target[0], teacher[0])
    assert torch.equal(target[1], parent[1])
