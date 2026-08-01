from __future__ import annotations

import torch

import scripts.collect_vq2_lc129_phase8_9_expanded_rescue_corpus as lc129


class FakeActor:
    def initial_state(self, batch_size, *, device, dtype=torch.float32):
        return torch.zeros(1, batch_size, 3, device=device, dtype=dtype)

    def forward_step(self, observation, state):
        rows = observation.shape[0]
        from pufferlib.vq2_recurrent import RecurrentActorOutput
        values = observation[:, :4]
        return RecurrentActorOutput(values, values + 1, values + 2), state + 1


def test_lc129_source_lock_and_split_actor_contract() -> None:
    payload = lc129.verify_inputs()
    assert payload["numerically_admitted"]
    actor = lc129.SplitBatchActor((FakeActor(), FakeActor()))
    state = actor.initial_state(lc129.TOTAL_AGENTS, device="cpu")
    observation = torch.arange(
        lc129.TOTAL_AGENTS * 4, dtype=torch.float32
    ).reshape(lc129.TOTAL_AGENTS, 4)
    output, next_state = actor.forward_step(observation, state)
    assert output.mean.shape == (lc129.TOTAL_AGENTS, 4)
    assert torch.equal(output.mean, observation)
    assert next_state.shape == state.shape
    assert torch.equal(next_state, state + 1)


def test_lc129_seed_ranges_do_not_overlap() -> None:
    original_max = 255
    new_min = lc129.GROUP_SIZE + lc129.NEW_RELATIVE_SEED_MIN
    assert original_max < new_min
    assert lc129.ACTOR_BATCH_SIZE * 2 == lc129.TOTAL_AGENTS
