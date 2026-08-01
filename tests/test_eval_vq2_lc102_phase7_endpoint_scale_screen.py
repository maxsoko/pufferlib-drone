from __future__ import annotations

import torch

import scripts.eval_vq2_lc102_phase7_endpoint_scale_screen as lc102


def test_lc102_source_lock_and_scale_contract() -> None:
    parent = lc102.verify_inputs()
    assert lc102.TARGET_PHASE == 7
    assert lc102.TARGET_RAW_INDEX == 8
    assert lc102.GROUP_SIZE == 64
    assert lc102.ALPHAS == (0.0, 0.25, 0.50, 0.75, 1.0, 1.50, 2.0)
    assert lc102.GROUP_SIZE * len(lc102.ALPHAS) == 448
    baseline = lc102.candidate_state_for_index(parent["model_state"], 0)
    assert all(
        torch.equal(baseline[name], value)
        for name, value in parent["model_state"].items()
    )
    endpoint = lc102.endpoint_decoder()
    exact = lc102.candidate_state_for_index(
        parent["model_state"], lc102.ALPHAS.index(1.0)
    )
    assert torch.equal(
        exact["indexed_phase_residual_output"][lc102.TARGET_PHASE],
        endpoint["output_weight"],
    )
    assert torch.equal(
        exact["indexed_phase_residual_output_bias"][lc102.TARGET_PHASE],
        endpoint["output_bias"],
    )


def test_lc102_full_batch_execution_selects_complete_actor_vectors() -> None:
    class Output:
        def __init__(self, value: float):
            self.mean = torch.full((448, 4), value)

    class Actor:
        def __init__(self, value: float):
            self.value = value

        def forward_step(self, actor_input, recurrent):
            assert actor_input.shape[0] == 448
            return Output(self.value), recurrent + 1.0

    execution = {
        "actors": [Actor(float(index)) for index in range(len(lc102.ALPHAS))],
        "recurrent": [torch.zeros((1, 448, 2)) for _ in lc102.ALPHAS],
    }
    actions = lc102.execute_actor_actions(
        execution, torch.zeros((448, 3)), torch.ones(448, dtype=torch.bool),
        torch.zeros((448, 1)), None,
    )
    for group in range(len(lc102.ALPHAS)):
        assert torch.equal(
            actions[lc102.group_slice(group)],
            torch.full((lc102.GROUP_SIZE, 4), float(group)),
        )
