import sys

import numpy as np


def test_native_rollout_trace_exposes_fp32_rollout_and_initial_state():
    from pufferlib import _C, pufferl

    saved_argv = sys.argv
    try:
        sys.argv = [
            "native-rollout-test",
            "--vec.total-agents", "8",
            "--vec.num-buffers", "1",
            "--vec.num-threads", "1",
            "--train.horizon", "4",
            "--train.minibatch-size", "32",
        ]
        cfg = pufferl.load_config("drone_race_full_policy_official_fit")
    finally:
        sys.argv = saved_argv

    runner = _C.create_pufferl(cfg)
    try:
        _C.rollouts(runner)
        trace = _C.rollout_trace(runner)
    finally:
        _C.close(runner)

    assert trace["observations"].shape == (4, 8, 32)
    assert trace["actions"].shape == (4, 8, 4)
    assert trace["terminals"].shape == (4, 8)
    assert trace["initial_states"].shape == (3, 8, 128)
    assert all(value.dtype == np.float32 for value in trace.values())
    assert np.all(trace["initial_states"] == 0.0)
    assert np.isfinite(trace["actions"]).all()
