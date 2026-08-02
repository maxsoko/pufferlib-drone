from __future__ import annotations

import torch

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_public_phase import OFFICIAL_PROGRESS_SCALE
from pufferlib.vq2_recurrent_phase_residual import VQ2PhaseActionSequenceActor
import scripts.build_vq2_lc213_phase16_17_action_sequence as target


def test_lc213_source_lock_and_sequence() -> None:
    target.verify_inputs()
    sequence, report = target.extract_action_sequence()
    assert sequence.shape == (target.SEQUENCE_LENGTH, 4)
    assert report["duplicate_action_max_error"] == 0.0
    assert report["phase16_steps"] == 1_392
    assert report["phase17_steps"] == 937


def test_action_sequence_actor_counter_and_scope() -> None:
    parent = target.verify_inputs()
    sequence = torch.tensor([
        [0.1, 0.2, 0.3, 0.0],
        [0.4, 0.5, 0.6, 0.0],
    ])
    contract = parent["model"]
    actor = VQ2PhaseActionSequenceActor(
        target_phase=int(contract["adapter_target_phase"]),
        sequence_phase_min=16,
        sequence_phase_max_exclusive=18,
        sequence_length=2,
        hidden_size=int(contract["hidden_size"]),
        residual_size=int(contract["residual_size"]),
        adapter_size=int(contract["adapter_size"]),
        initial_std=float(contract["initial_std"]),
    )
    actor.load_phase_local_state(parent["model_state"], sequence)
    observation = torch.zeros(1, LEGAL_OBS_SIZE + 1)
    state = actor.initial_state(1, device="cpu")
    observation[:, LEGAL_OBS_SIZE] = 15 / OFFICIAL_PROGRESS_SCALE
    base, state = actor.forward_step(observation, state)
    assert state[0, 0, -1] == 0
    observation[:, LEGAL_OBS_SIZE] = 16 / OFFICIAL_PROGRESS_SCALE
    first, state = actor.forward_step(observation, state)
    second, state = actor.forward_step(observation, state)
    assert torch.equal(first.mean[0], sequence[0])
    assert torch.equal(second.mean[0], sequence[1])
    assert state[0, 0, -1] == 2
    observation[:, LEGAL_OBS_SIZE] = 18 / OFFICIAL_PROGRESS_SCALE
    after, state = actor.forward_step(observation, state)
    assert state[0, 0, -1] == 2
    assert not torch.equal(after.mean[0], sequence[1])
    assert base.mean.shape == after.mean.shape == (1, 4)
