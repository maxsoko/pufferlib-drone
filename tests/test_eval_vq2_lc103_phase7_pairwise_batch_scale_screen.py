from __future__ import annotations

import torch

import scripts.eval_vq2_lc103_phase7_pairwise_batch_scale_screen as lc103


def test_lc103_source_lock_and_pair_contract() -> None:
    lc103.configure()
    lc103.verify_inputs()
    assert lc103.GROUP_SIZE == 128
    assert lc103.PAIR_SIZE == 256
    assert lc103.ALPHAS == (0.0, 0.50, 1.0, 2.0)
    assert lc103.GROUP_SIZE * len(lc103.ALPHAS) == 512


def test_lc103_pair_indices_keep_256_row_actor_batches() -> None:
    for group in range(len(lc103.ALPHAS)):
        indices = lc103.pair_indices(group, device=torch.device("cpu"))
        assert indices.shape == (256,)
        assert torch.equal(indices[:128], torch.arange(128))
        assert torch.equal(
            indices[128:], torch.arange(group * 128, (group + 1) * 128)
        )


def test_lc103_execution_selects_complete_pair_vectors() -> None:
    lc103.configure()

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
        "actors": [Actor(float(index)) for index in range(len(lc103.ALPHAS))],
        "recurrent": [torch.zeros((1, 256, 2)) for _ in lc103.ALPHAS],
        "indices": [
            lc103.pair_indices(group, device=torch.device("cpu"))
            for group in range(len(lc103.ALPHAS))
        ],
    }
    actions = lc103.execute_actor_actions(
        execution, torch.zeros((512, 3)), torch.ones(512, dtype=torch.bool),
        torch.zeros((512, 1)), None,
    )
    for group in range(len(lc103.ALPHAS)):
        assert torch.equal(
            actions[lc103.group_slice(group)],
            torch.full((128, 4), float(group)),
        )
