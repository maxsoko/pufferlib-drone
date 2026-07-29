"""Public-phase recurrent actor with an exact-zero joint action residual."""

from __future__ import annotations

import math

import torch
from torch import nn

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, VQ2VisualEncoder
from pufferlib.vq2_recurrent import ACTION_SIZE, RecurrentActorOutput
from pufferlib.vq2_recurrent_phase import PHASE_LEGAL_OBS_SIZE


class VQ2PhaseResidualActor(nn.Module):
    """One recurrent actor with a phase-gated, joint four-action correction."""

    def __init__(self, *, hidden_size: int = 256, initial_std: float = 0.15) -> None:
        super().__init__()
        if hidden_size <= 0 or initial_std <= 0.0:
            raise ValueError("hidden size and initial standard deviation must be positive")
        self.hidden_size = hidden_size
        self.action_size = ACTION_SIZE
        self.encoder = VQ2VisualEncoder(hidden_size)
        self.phase_embedding = nn.Linear(1, hidden_size, bias=False)
        self.recurrent = nn.GRU(
            hidden_size, hidden_size, num_layers=1, batch_first=True
        )
        self.action_head = nn.Linear(hidden_size, ACTION_SIZE)
        self.phase_action_residual = nn.Linear(hidden_size, ACTION_SIZE, bias=False)
        self.log_std = nn.Parameter(
            torch.full((ACTION_SIZE,), math.log(initial_std), dtype=torch.float32)
        )
        nn.init.zeros_(self.phase_embedding.weight)
        nn.init.zeros_(self.phase_action_residual.weight)

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

    def load_phase_zero_base_state(self, base_state: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(base_state, strict=False)
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != {
            "phase_embedding.weight",
            "phase_action_residual.weight",
        }:
            raise RuntimeError("base actor differs outside the public-phase paths")
        nn.init.zeros_(self.phase_embedding.weight)
        nn.init.zeros_(self.phase_action_residual.weight)

    def forward_sequence(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 3 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("phase-residual actor accepts [batch,time,4119] only")
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        if tuple(state.shape) != (1, batch, self.hidden_size):
            raise ValueError("phase-residual recurrent state shape changed")
        legal = observation[..., :LEGAL_OBS_SIZE]
        phase = observation[..., LEGAL_OBS_SIZE:]
        if not bool(torch.isfinite(phase).all()) or bool(
            ((phase < -1e-6) | (phase > 1.0 + 1e-6)).any()
        ):
            raise ValueError("public gate phase must be finite and in [0,1]")
        encoded = self.encoder(legal) + self.phase_embedding(phase)
        recurrent, next_state = self.recurrent(encoded, state)
        base = self.action_head(recurrent)
        residual = phase * self.phase_action_residual(recurrent)
        pre_tanh_mean = base + residual
        mean = torch.tanh(pre_tanh_mean)
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        return RecurrentActorOutput(mean, pre_tanh_mean, log_std), next_state

    def forward_step(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("phase-residual actor step accepts [batch,4119] only")
        output, next_state = self.forward_sequence(observation.unsqueeze(1), state)
        return RecurrentActorOutput(
            output.mean[:, 0], output.pre_tanh_mean[:, 0], output.log_std[:, 0]
        ), next_state

