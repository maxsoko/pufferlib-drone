from __future__ import annotations

import torch

import scripts.eval_vq2_lc094_full_batch_serialized_checkpoint as lc094


def test_lc094_source_locks_partitioned_rejection() -> None:
    lc094.verify_inputs()


def test_lc094_full_batch_execution_selects_complete_actor_vectors() -> None:
    class Output:
        def __init__(self, value: float):
            self.mean = torch.full((256, 4), value)

    class Actor:
        def __init__(self, value: float):
            self.value = value

        def forward_step(self, actor_input, recurrent):
            assert actor_input.shape[0] == 256
            return Output(self.value), recurrent + 1.0

    execution = {
        "actors": (Actor(1.0), Actor(2.0)),
        "recurrent": [torch.zeros((1, 256, 2)), torch.zeros((1, 256, 2))],
    }
    actions = lc094.execute_actor_actions(
        execution,
        torch.zeros((256, 3)),
        torch.ones(256, dtype=torch.bool),
        torch.zeros((256, 1)),
        None,
    )
    assert torch.equal(actions[:128], torch.ones((128, 4)))
    assert torch.equal(actions[128:], torch.full((128, 4), 2.0))
    assert all(torch.equal(state, torch.ones_like(state)) for state in execution["recurrent"])
