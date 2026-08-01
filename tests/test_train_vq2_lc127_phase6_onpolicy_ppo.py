from __future__ import annotations

import numpy as np
import torch

import scripts.train_vq2_lc127_phase6_onpolicy_ppo as lc127


def test_lc127_source_lock_and_log_probability() -> None:
    payload = lc127.verify_inputs()
    assert payload["numerically_admitted"]
    mean = torch.zeros(3, 4)
    sample = torch.zeros(3, 4)
    log_probability = lc127.normal_log_probability(sample, mean, 0.01)
    assert log_probability.shape == (3,)
    assert torch.isfinite(log_probability).all()
    assert torch.equal(log_probability, log_probability[:1].expand_as(log_probability))


def test_lc127_trajectory_advantage_is_centered() -> None:
    maximum = np.zeros(lc127.TOTAL_AGENTS, dtype=np.int32)
    passed = np.zeros(lc127.TOTAL_AGENTS, dtype=bool)
    maximum[[2, 4, 8]] = [6, 7, 10]
    passed[8] = True
    agents = np.asarray([2, 2, 4, 8, 8, 8], dtype=np.int64)
    advantage, metrics = lc127.trajectory_advantages(maximum, passed, agents)
    assert abs(float(advantage[[2, 4, 8]].mean())) < 1e-6
    assert metrics["queried_score_max"] == 14.0
    assert advantage[8] > advantage[4] > advantage[2]


def test_lc127_optional_phase_return_breaks_sparse_tie(monkeypatch) -> None:
    monkeypatch.setattr(lc127, "PHASE_RETURN_ADVANTAGE_WEIGHT", 1.0)
    maximum = np.zeros(lc127.TOTAL_AGENTS, dtype=np.int32)
    maximum[[2, 4, 8]] = 8
    passed = np.zeros(lc127.TOTAL_AGENTS, dtype=bool)
    phase_return = np.zeros(lc127.TOTAL_AGENTS, dtype=np.float64)
    phase_return[[2, 4, 8]] = [-3.0, 1.0, 9.0]
    agents = np.asarray([2, 2, 4, 8, 8, 8], dtype=np.int64)
    advantage, metrics = lc127.trajectory_advantages(
        maximum, passed, agents, phase_return
    )
    assert abs(float(advantage[[2, 4, 8]].mean())) < 1e-6
    assert advantage[8] > advantage[4] > advantage[2]
    assert metrics["queried_phase_return_std"] > 0.0


def test_lc127_only_phase6_is_trainable() -> None:
    payload = lc127.verify_inputs()
    state = payload["model_state"]
    assert lc127.PARAMETER_NAMES == (
        "indexed_phase_residual_input",
        "indexed_phase_residual_input_bias",
        "indexed_phase_residual_output",
        "indexed_phase_residual_output_bias",
    )
    assert all(state[name].shape[0] > lc127.TARGET_PHASE for name in lc127.PARAMETER_NAMES)
