from __future__ import annotations

import numpy as np
import pytest
import torch

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE
from scripts.collect_vq2_public_phase_dagger_next import (
    AGENTS,
    phase_actor_observation,
)


def test_phase_actor_observation_appends_only_held_public_value() -> None:
    native = torch.zeros(AGENTS, ENV_OBS_SIZE)
    native[:, LEGAL_OBS_SIZE:] = 99.0
    held = np.full(AGENTS, np.float32(1.0 / 6.0), dtype=np.float32)
    actor = phase_actor_observation(native, held, device=torch.device("cpu"))
    assert actor.shape == (AGENTS, LEGAL_OBS_SIZE + 1)
    assert torch.all(actor[:, :LEGAL_OBS_SIZE] == 0.0)
    assert torch.allclose(actor[:, -1], torch.full((AGENTS,), 1.0 / 6.0))


def test_phase_actor_observation_rejects_wrong_native_shape() -> None:
    with pytest.raises(ValueError, match="batch changed"):
        phase_actor_observation(
            torch.zeros(1, ENV_OBS_SIZE),
            np.zeros(1, dtype=np.float32),
            device=torch.device("cpu"),
        )

