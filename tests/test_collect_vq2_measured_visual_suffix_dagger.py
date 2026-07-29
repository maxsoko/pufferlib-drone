from __future__ import annotations

import numpy as np
import pytest

from pufferlib.vq2_informed import MASK_SIZE
from scripts import collect_vq2_measured_visual_suffix_dagger as dagger


def test_alias_group_changes_only_trailing_masks_during_hold():
    observation = np.zeros((4, MASK_SIZE + dagger.PHASE_TAIL_SIZE), dtype=np.float32)
    observation[:, 20 * 64 + 30 : 20 * 64 + 33] = 1.0
    observation[:, MASK_SIZE:] = np.arange(dagger.PHASE_TAIL_SIZE, dtype=np.float32)
    result = dagger.apply_fixed_alias_group(
        observation, local_step=0, alias_start_agent=2
    )
    assert np.array_equal(result[:2], observation[:2])
    assert result[2:, :MASK_SIZE].sum() > observation[2:, :MASK_SIZE].sum()
    assert np.array_equal(result[:, MASK_SIZE:], observation[:, MASK_SIZE:])
    after = dagger.apply_fixed_alias_group(
        observation, local_step=dagger.ALIAS_HOLD_STEPS, alias_start_agent=2
    )
    assert np.array_equal(after, observation)


def test_storage_parts_round_trip_and_excludes_privilege():
    observation = np.zeros((2, MASK_SIZE + dagger.PHASE_TAIL_SIZE), dtype=np.float32)
    observation[0, 7] = 0.5
    observation[:, MASK_SIZE:] = 0.25
    mask, tail = dagger.storage_parts(observation)
    assert mask.shape == (2, MASK_SIZE)
    assert tail.shape == (2, dagger.PHASE_TAIL_SIZE)
    assert mask[0, 7] == 128
    assert tail == pytest.approx(0.25)


def test_collection_admission_requires_one_terminal_per_episode():
    lengths = np.full(dagger.AGENTS, 10, dtype=np.int32)
    terminals = np.ones(dagger.AGENTS, dtype=np.int32)
    terminal_last = np.ones(dagger.AGENTS, dtype=bool)
    kwargs = dict(
        lengths=lengths,
        terminal_count=terminals,
        terminal_is_last=terminal_last,
        records=int(lengths.sum()),
        n294_replay_error=0.0,
        suffix_replay_error=0.0,
        executed_action_error=0.0,
        action_envelope_violations=0,
        nonfinite_action=False,
        phase_decreases=0,
    )
    assert dagger.collection_admitted(**kwargs)
    kwargs["terminal_count"] = np.zeros(dagger.AGENTS, dtype=np.int32)
    assert not dagger.collection_admitted(**kwargs)
