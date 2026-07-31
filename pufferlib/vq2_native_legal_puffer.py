"""Deployable PyTorch view of a native CUDA PuffeRL VQ2 checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pufferlib.vq2_informed import ENV_OBS_SIZE, LEGAL_OBS_SIZE
from pufferlib.vq2_recurrent import ACTION_SIZE, RecurrentActorOutput
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE


class VQ2NativeLegalPufferActor(nn.Module):
    """One-linear-encoder/MinGRU actor with no privileged input surface.

    Native training keeps a 4,152-wide CUDA tensor for ABI compatibility, but
    LC016 zeros its 34-value tail. This deployment view drops those columns and
    maps held public progress into the legal slot used during training.
    """

    def __init__(self, *, hidden_size: int = 256) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden size must be positive")
        self.hidden_size = hidden_size
        self.encoder_weight = nn.Parameter(
            torch.empty(hidden_size, LEGAL_OBS_SIZE), requires_grad=False
        )
        self.decoder_weight = nn.Parameter(
            torch.empty(ACTION_SIZE + 1, hidden_size), requires_grad=False
        )
        self.log_std = nn.Parameter(
            torch.empty(ACTION_SIZE), requires_grad=False
        )
        self.recurrent_weight = nn.Parameter(
            torch.empty(3 * hidden_size, hidden_size), requires_grad=False
        )

    @property
    def native_parameter_count(self) -> int:
        return (
            self.hidden_size * ENV_OBS_SIZE
            + (ACTION_SIZE + 1) * self.hidden_size
            + ACTION_SIZE
            + 3 * self.hidden_size * self.hidden_size
        )

    @classmethod
    def from_flat_weights(
        cls, values: np.ndarray | torch.Tensor, *, hidden_size: int = 256,
    ) -> "VQ2NativeLegalPufferActor":
        flat = torch.as_tensor(values, dtype=torch.float32).flatten().clone()
        model = cls(hidden_size=hidden_size)
        if flat.numel() != model.native_parameter_count:
            raise ValueError(
                f"native checkpoint has {flat.numel()} floats, expected "
                f"{model.native_parameter_count}"
            )
        cursor = 0
        native_encoder_count = hidden_size * ENV_OBS_SIZE
        native_encoder = flat[cursor : cursor + native_encoder_count].reshape(
            hidden_size, ENV_OBS_SIZE
        )
        cursor += native_encoder_count
        decoder_count = (ACTION_SIZE + 1) * hidden_size
        decoder = flat[cursor : cursor + decoder_count].reshape(
            ACTION_SIZE + 1, hidden_size
        )
        cursor += decoder_count
        log_std = flat[cursor : cursor + ACTION_SIZE]
        cursor += ACTION_SIZE
        recurrent_count = 3 * hidden_size * hidden_size
        recurrent = flat[cursor : cursor + recurrent_count].reshape(
            3 * hidden_size, hidden_size
        )
        cursor += recurrent_count
        if cursor != flat.numel():
            raise RuntimeError("native checkpoint layout accounting changed")
        with torch.no_grad():
            model.encoder_weight.copy_(native_encoder[:, :LEGAL_OBS_SIZE])
            model.decoder_weight.copy_(decoder)
            model.log_std.copy_(log_std)
            model.recurrent_weight.copy_(recurrent)
        return model

    @classmethod
    def from_binary(
        cls, path: Path | str, *, hidden_size: int = 256,
    ) -> "VQ2NativeLegalPufferActor":
        values = np.fromfile(Path(path), dtype="<f4")
        return cls.from_flat_weights(values, hidden_size=hidden_size)

    def initial_state(
        self, batch_size: int, *, device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        return torch.zeros(
            1, batch_size, self.hidden_size, device=device, dtype=dtype
        )

    @staticmethod
    def _g(values: torch.Tensor) -> torch.Tensor:
        return torch.where(values >= 0, values + 0.5, values.sigmoid())

    def _forward_compact_step(
        self, compact: torch.Tensor, state: torch.Tensor,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        encoded = F.linear(compact, self.encoder_weight)
        hidden, gate, projection = F.linear(
            encoded, self.recurrent_weight
        ).chunk(3, dim=-1)
        recurrent = torch.lerp(state[0], self._g(hidden), gate.sigmoid())
        feature = projection.sigmoid() * recurrent + (
            1.0 - projection.sigmoid()
        ) * encoded
        decoded = F.linear(feature, self.decoder_weight)
        mean = decoded[:, :ACTION_SIZE]
        log_std = self.log_std.expand_as(mean)
        return RecurrentActorOutput(mean, mean, log_std), recurrent.unsqueeze(0)

    def forward_step(
        self, observation: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("native legal Puffer accepts [batch,4119] only")
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        if tuple(state.shape) != (1, batch, self.hidden_size):
            raise ValueError("native legal Puffer recurrent state shape changed")
        compact = observation[:, :LEGAL_OBS_SIZE].clone()
        compact[:, LEGAL_OBS_SIZE - 1] = observation[:, LEGAL_OBS_SIZE]
        return self._forward_compact_step(compact, state)

    def forward_sequence(
        self, observation: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 3 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("native legal Puffer accepts [batch,time,4119] only")
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        means: list[torch.Tensor] = []
        log_stds: list[torch.Tensor] = []
        for step in range(observation.shape[1]):
            output, state = self.forward_step(observation[:, step], state)
            means.append(output.mean)
            log_stds.append(output.log_std)
        mean = torch.stack(means, dim=1)
        log_std = torch.stack(log_stds, dim=1)
        return RecurrentActorOutput(mean, mean, log_std), state
