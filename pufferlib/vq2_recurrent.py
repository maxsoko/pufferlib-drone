"""Small competition-legal recurrent actor for VQ2.

The module accepts exactly the 4,118-value public observation ABI.  It has one
visual encoder, one GRU, and one joint four-channel action head; no critic,
teacher, course phase, or privileged decoder is part of the actor artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, VQ2VisualEncoder


ACTION_SIZE = 4


@dataclass(frozen=True)
class RecurrentActorOutput:
    mean: torch.Tensor
    pre_tanh_mean: torch.Tensor
    log_std: torch.Tensor


class VQ2RecurrentActor(nn.Module):
    """A 256-wide CNN/GRU Puffer actor over the complete normalized CTBR vector."""

    def __init__(
        self,
        *,
        hidden_size: int = 256,
        initial_std: float = 0.15,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or initial_std <= 0.0:
            raise ValueError("hidden size and initial standard deviation must be positive")
        self.hidden_size = hidden_size
        self.action_size = ACTION_SIZE
        self.encoder = VQ2VisualEncoder(hidden_size)
        self.recurrent = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.action_head = nn.Linear(hidden_size, ACTION_SIZE)
        self.log_std = nn.Parameter(
            torch.full((ACTION_SIZE,), math.log(initial_std), dtype=torch.float32)
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        return torch.zeros(1, batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward_sequence(
        self,
        legal_observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if legal_observation.ndim != 3 or legal_observation.shape[-1] != LEGAL_OBS_SIZE:
            raise ValueError(
                "VQ2RecurrentActor accepts [batch,time,4118] legal observations only"
            )
        batch = legal_observation.shape[0]
        if state is None:
            state = self.initial_state(
                batch,
                device=legal_observation.device,
                dtype=legal_observation.dtype,
            )
        expected_state = (1, batch, self.hidden_size)
        if tuple(state.shape) != expected_state:
            raise ValueError(
                f"recurrent state shape {tuple(state.shape)} != {expected_state}"
            )
        encoded = self.encoder(legal_observation)
        recurrent, next_state = self.recurrent(encoded, state)
        pre_tanh_mean = self.action_head(recurrent)
        mean = torch.tanh(pre_tanh_mean)
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        return RecurrentActorOutput(mean, pre_tanh_mean, log_std), next_state

    def forward_step(
        self,
        legal_observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if legal_observation.ndim != 2 or legal_observation.shape[-1] != LEGAL_OBS_SIZE:
            raise ValueError(
                "VQ2RecurrentActor step accepts [batch,4118] legal observations only"
            )
        output, next_state = self.forward_sequence(
            legal_observation.unsqueeze(1), state
        )
        return RecurrentActorOutput(
            output.mean[:, 0],
            output.pre_tanh_mean[:, 0],
            output.log_std[:, 0],
        ), next_state

    @staticmethod
    def sample_action(
        output: RecurrentActorOutput,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(
            output.pre_tanh_mean.shape,
            device=output.pre_tanh_mean.device,
            dtype=output.pre_tanh_mean.dtype,
            generator=generator,
        )
        return torch.tanh(output.pre_tanh_mean + output.log_std.exp() * noise)

