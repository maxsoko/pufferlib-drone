from __future__ import annotations

import numpy as np
import pytest

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE
from scripts.collect_vq2_public_phase_dagger import (
    PHASE_PRIVILEGED_INDEX,
    PHASE_TAIL_SIZE,
    STATUS_HOLD_STEPS,
    phase_storage_parts,
    update_held_phase,
)


def test_public_phase_is_sampled_and_held_at_four_hz() -> None:
    assert STATUS_HOLD_STEPS == 16
    held = np.zeros(2, dtype=np.float32)
    held, sampled = update_held_phase(
        np.asarray([0.0, 1.0 / 6.0], dtype=np.float32), held, step=0
    )
    assert sampled
    assert held.tolist() == [0.0, np.float32(1.0 / 6.0)]
    held, sampled = update_held_phase(
        np.asarray([1.0 / 6.0, 2.0 / 6.0], dtype=np.float32), held, step=1
    )
    assert not sampled
    assert held.tolist() == [0.0, np.float32(1.0 / 6.0)]


def test_phase_storage_appends_one_value_after_legacy_legal_tail() -> None:
    observation = np.zeros((2, ENV_OBS_SIZE), dtype=np.float32)
    observation[:, :4096] = 0.5
    phase = np.asarray([0.0, 0.5], dtype=np.float32)
    mask, tail = phase_storage_parts(observation, phase)
    assert mask.shape == (2, 4096)
    assert tail.shape == (2, PHASE_TAIL_SIZE)
    assert PHASE_LEGAL_OBS_SIZE == LEGAL_OBS_SIZE + 1
    assert np.array_equal(tail[:, -1], phase)


def test_phase_contract_uses_only_named_ordered_phase_target() -> None:
    assert LEGAL_OBS_SIZE <= PHASE_PRIVILEGED_INDEX < ENV_OBS_SIZE
    with pytest.raises(RuntimeError, match=r"\[0,1\]"):
        update_held_phase(
            np.asarray([1.1], dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            step=0,
        )

