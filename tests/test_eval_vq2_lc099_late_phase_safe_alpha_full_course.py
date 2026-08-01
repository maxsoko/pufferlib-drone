from __future__ import annotations

import torch

import scripts.eval_vq2_lc099_late_phase_safe_alpha_full_course as lc099


def test_lc099_source_locks_promoted_safe_checkpoint() -> None:
    lc099.verify_inputs()


def test_lc099_full_batch_execution_selects_complete_actor_vectors() -> None:
    class Output:
        def __init__(self, value: float):
            self.mean = torch.full((lc099.TOTAL_AGENTS, 4), value)

    class Actor:
        def __init__(self, value: float):
            self.value = value

        def forward_step(self, actor_input, recurrent):
            assert actor_input.shape[0] == lc099.TOTAL_AGENTS
            return Output(self.value), recurrent + 1.0

    execution = {
        "actors": (Actor(1.0), Actor(2.0)),
        "recurrent": [
            torch.zeros((1, lc099.TOTAL_AGENTS, 2)),
            torch.zeros((1, lc099.TOTAL_AGENTS, 2)),
        ],
    }
    actions = lc099.execute_actor_actions(
        execution,
        torch.zeros((lc099.TOTAL_AGENTS, 3)),
        torch.ones(lc099.TOTAL_AGENTS, dtype=torch.bool),
        torch.zeros((lc099.TOTAL_AGENTS, 1)),
        None,
    )
    assert torch.equal(actions[:lc099.GROUP_SIZE], torch.ones((lc099.GROUP_SIZE, 4)))
    assert torch.equal(
        actions[lc099.GROUP_SIZE:], torch.full((lc099.GROUP_SIZE, 4), 2.0)
    )
    assert all(
        torch.equal(state, torch.ones_like(state)) for state in execution["recurrent"]
    )


def test_lc099_requires_progress_and_no_safety_regression() -> None:
    parent = {
        "transport_pass": True, "mean_gates_passed": 3.0,
        "promotion_target_passes": 2, "crash_rate": 0.1,
        "maximum_raw_index": 7,
    }
    candidate = {
        "transport_pass": True,
        "mean_gates_passed": 3.0 + lc099.MINIMUM_MEAN_GATE_GAIN,
        "promotion_target_passes": 3, "crash_rate": 0.1,
        "maximum_raw_index": 7,
    }
    assert lc099.choose_candidate(parent, candidate)
    candidate["crash_rate"] = 0.11
    assert not lc099.choose_candidate(parent, candidate)
