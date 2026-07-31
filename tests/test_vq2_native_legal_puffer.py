from __future__ import annotations

import numpy as np
import pytest
import torch

from pufferlib.vq2_informed import LEGAL_OBS_SIZE
from pufferlib.vq2_native_legal_puffer import VQ2NativeLegalPufferActor
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE


def test_native_checkpoint_layout_and_public_progress_mapping() -> None:
    template = VQ2NativeLegalPufferActor()
    values = np.zeros(template.native_parameter_count, dtype=np.float32)
    model = VQ2NativeLegalPufferActor.from_flat_weights(values)
    observation = torch.zeros(2, PHASE_LEGAL_OBS_SIZE)
    observation[:, LEGAL_OBS_SIZE - 1] = 0.75
    observation[:, LEGAL_OBS_SIZE] = 3.0 / 6.0
    output, state = model.forward_step(observation)
    assert output.mean.shape == (2, 4)
    assert state.shape == (1, 2, 256)
    assert torch.count_nonzero(output.mean) == 0


def test_actor_rejects_native_privileged_width() -> None:
    model = VQ2NativeLegalPufferActor.from_flat_weights(
        np.zeros(VQ2NativeLegalPufferActor().native_parameter_count, np.float32)
    )
    with pytest.raises(ValueError, match="4119"):
        model.forward_step(torch.zeros(1, 4152))
