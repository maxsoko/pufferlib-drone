"""Recurrent VQ2 actor with one public official-progress scalar.

The first 4,118 values retain the frozen legal camera/IMU/actuator/action/timing
ABI. The final value is held ``active_gate_index / 6`` from public race status.
No state estimate, gate geometry, or privileged decoder enters this module.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, VQ2VisualEncoder
from pufferlib.vq2_recurrent import ACTION_SIZE, RecurrentActorOutput


PUBLIC_PHASE_SIZE = 1
PHASE_LEGAL_OBS_SIZE = LEGAL_OBS_SIZE + PUBLIC_PHASE_SIZE
OFFICIAL_GATE_COUNT = 6


class VQ2PhaseRecurrentActor(nn.Module):
    """One 256-wide recurrent actor over legal perception plus public phase."""

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
        self.phase_embedding = nn.Linear(PUBLIC_PHASE_SIZE, hidden_size, bias=False)
        nn.init.zeros_(self.phase_embedding.weight)
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

    def load_phase_zero_base_state(self, base_state: dict[str, torch.Tensor]) -> None:
        """Transfer a 4,118-input actor exactly and zero the new phase path."""

        incompatible = self.load_state_dict(base_state, strict=False)
        if incompatible.unexpected_keys or incompatible.missing_keys != [
            "phase_embedding.weight"
        ]:
            raise RuntimeError(
                "base actor state differs outside the single phase embedding"
            )
        nn.init.zeros_(self.phase_embedding.weight)

    def forward_sequence(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 3 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError(
                "VQ2PhaseRecurrentActor accepts [batch,time,4119] observations only"
            )
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        expected_state = (1, batch, self.hidden_size)
        if tuple(state.shape) != expected_state:
            raise ValueError(
                f"recurrent state shape {tuple(state.shape)} != {expected_state}"
            )
        legal = observation[..., :LEGAL_OBS_SIZE]
        phase = observation[..., LEGAL_OBS_SIZE:]
        if not bool(torch.isfinite(phase).all()):
            raise ValueError("public gate phase must be finite")
        if bool(((phase < -1e-6) | (phase > 1.0 + 1e-6)).any()):
            raise ValueError("public gate phase escaped [0,1]")
        encoded = self.encoder(legal) + self.phase_embedding(phase)
        recurrent, next_state = self.recurrent(encoded, state)
        pre_tanh_mean = self.action_head(recurrent)
        mean = torch.tanh(pre_tanh_mean)
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        return RecurrentActorOutput(mean, pre_tanh_mean, log_std), next_state

    def forward_step(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError(
                "VQ2PhaseRecurrentActor step accepts [batch,4119] observations only"
            )
        output, next_state = self.forward_sequence(observation.unsqueeze(1), state)
        return RecurrentActorOutput(
            output.mean[:, 0],
            output.pre_tanh_mean[:, 0],
            output.log_std[:, 0],
        ), next_state

