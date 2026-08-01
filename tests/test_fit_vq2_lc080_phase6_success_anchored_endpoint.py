from __future__ import annotations

import numpy as np
import torch

import scripts.fit_vq2_lc080_phase6_success_anchored_endpoint as lc080


def test_lc080_source_lock_and_contract() -> None:
    lc080.verify_inputs()
    assert lc080.TARGET_PHASE == 6
    assert lc080.MINIMUM_FAILURE_IMPROVEMENT == 1.01
    assert lc080.MAXIMUM_SUCCESS_ACTION_DRIFT_MSE == 0.00025
    assert lc080.ALPHAS[0] == 0.0
    assert lc080.ALPHAS[-1] == 1.0


def test_lc080_split_holds_one_success_in_each_partition() -> None:
    present = np.arange(35, dtype=np.int64)
    outcome = np.zeros(256, dtype=np.int8)
    outcome[present[-2:]] = 1
    train, validation = lc080.stratified_agent_split(present, outcome)
    assert int((outcome[train] == 1).sum()) == 1
    assert int((outcome[validation] == 1).sum()) == 1
    assert not np.intersect1d(train, validation).size
    assert np.array_equal(np.sort(np.concatenate((train, validation))), present)


def test_lc080_interpolation_endpoints() -> None:
    pw = torch.tensor([[1.0, 2.0]])
    pb = torch.tensor([3.0])
    ew = torch.tensor([[5.0, 6.0]])
    eb = torch.tensor([7.0])
    w0, b0 = lc080.interpolate_endpoint(pw, pb, ew, eb, 0.0)
    w1, b1 = lc080.interpolate_endpoint(pw, pb, ew, eb, 1.0)
    assert torch.equal(w0, pw) and torch.equal(b0, pb)
    assert torch.equal(w1, ew) and torch.equal(b1, eb)
