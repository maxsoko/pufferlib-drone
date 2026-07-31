"""Public-phase recurrent actor with an exact-zero joint action residual."""

from __future__ import annotations

import math

import torch
from torch import nn

from pufferlib.vq2_informed import LEGAL_OBS_SIZE, VQ2VisualEncoder
from pufferlib.vq2_recurrent import ACTION_SIZE, RecurrentActorOutput
from pufferlib.vq2_recurrent_phase import (
    OFFICIAL_GATE_COUNT,
    PHASE_LEGAL_OBS_SIZE,
    VQ2PhaseRecurrentActor,
)


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


class VQ2IndexedPhaseResidualActor(VQ2PhaseRecurrentActor):
    """Base phase actor plus one learned hidden-state residual per public index."""

    def __init__(self, *, hidden_size: int = 256, initial_std: float = 0.15) -> None:
        super().__init__(hidden_size=hidden_size, initial_std=initial_std)
        self.indexed_phase_action_residual = nn.Parameter(
            torch.zeros(OFFICIAL_GATE_COUNT + 1, ACTION_SIZE, hidden_size)
        )

    def load_base_state(self, base_state: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(base_state, strict=False)
        if incompatible.unexpected_keys or incompatible.missing_keys != [
            "indexed_phase_action_residual"
        ]:
            raise RuntimeError("base actor differs outside the indexed residual")
        nn.init.zeros_(self.indexed_phase_action_residual)

    def forward_sequence(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 3 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("indexed-phase actor accepts [batch,time,4119] only")
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        if tuple(state.shape) != (1, batch, self.hidden_size):
            raise ValueError("indexed-phase recurrent state shape changed")
        legal = observation[..., :LEGAL_OBS_SIZE]
        phase = observation[..., LEGAL_OBS_SIZE:]
        if not bool(torch.isfinite(phase).all()) or bool(
            ((phase < -1e-6) | (phase > 1.0 + 1e-6)).any()
        ):
            raise ValueError("public gate phase must be finite and in [0,1]")
        encoded = self.encoder(legal) + self.phase_embedding(phase)
        recurrent, next_state = self.recurrent(encoded, state)
        phase_index = torch.round(phase[..., 0] * OFFICIAL_GATE_COUNT)
        phase_index = phase_index.to(torch.long).clamp_(0, OFFICIAL_GATE_COUNT)
        selected = self.indexed_phase_action_residual[phase_index]
        residual = torch.einsum("bth,btoh->bto", recurrent, selected)
        pre_tanh_mean = self.action_head(recurrent) + residual
        mean = torch.tanh(pre_tanh_mean)
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        return RecurrentActorOutput(mean, pre_tanh_mean, log_std), next_state


class VQ2IndexedPhaseMLPResidualActor(VQ2PhaseRecurrentActor):
    """Base phase actor plus a small nonlinear residual per public index."""

    def __init__(
        self,
        *,
        hidden_size: int = 256,
        residual_size: int = 64,
        initial_std: float = 0.15,
    ) -> None:
        super().__init__(hidden_size=hidden_size, initial_std=initial_std)
        if residual_size <= 0:
            raise ValueError("residual size must be positive")
        self.residual_size = residual_size
        phases = OFFICIAL_GATE_COUNT + 1
        self.indexed_phase_residual_input = nn.Parameter(
            torch.empty(phases, residual_size, hidden_size)
        )
        self.indexed_phase_residual_input_bias = nn.Parameter(
            torch.zeros(phases, residual_size)
        )
        self.indexed_phase_residual_output = nn.Parameter(
            torch.zeros(phases, ACTION_SIZE, residual_size)
        )
        self.indexed_phase_residual_output_bias = nn.Parameter(
            torch.zeros(phases, ACTION_SIZE)
        )
        for phase_index in range(phases):
            nn.init.kaiming_uniform_(
                self.indexed_phase_residual_input[phase_index], a=math.sqrt(5)
            )

    def load_base_state(self, base_state: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(base_state, strict=False)
        expected = {
            "indexed_phase_residual_input",
            "indexed_phase_residual_input_bias",
            "indexed_phase_residual_output",
            "indexed_phase_residual_output_bias",
        }
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected:
            raise RuntimeError("base actor differs outside the indexed MLP residual")
        nn.init.zeros_(self.indexed_phase_residual_input_bias)
        nn.init.zeros_(self.indexed_phase_residual_output)
        nn.init.zeros_(self.indexed_phase_residual_output_bias)

    def forward_sequence(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 3 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("indexed-phase MLP actor accepts [batch,time,4119] only")
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        if tuple(state.shape) != (1, batch, self.hidden_size):
            raise ValueError("indexed-phase MLP recurrent state shape changed")
        legal = observation[..., :LEGAL_OBS_SIZE]
        phase = observation[..., LEGAL_OBS_SIZE:]
        if not bool(torch.isfinite(phase).all()) or bool(
            ((phase < -1e-6) | (phase > 1.0 + 1e-6)).any()
        ):
            raise ValueError("public gate phase must be finite and in [0,1]")
        encoded = self.encoder(legal) + self.phase_embedding(phase)
        recurrent, next_state = self.recurrent(encoded, state)
        phase_index = torch.round(phase[..., 0] * OFFICIAL_GATE_COUNT)
        phase_index = phase_index.to(torch.long).clamp_(0, OFFICIAL_GATE_COUNT)
        selected_input = self.indexed_phase_residual_input[phase_index]
        selected_input_bias = self.indexed_phase_residual_input_bias[phase_index]
        feature = torch.tanh(
            torch.einsum("bth,btrh->btr", recurrent, selected_input)
            + selected_input_bias
        )
        selected_output = self.indexed_phase_residual_output[phase_index]
        selected_output_bias = self.indexed_phase_residual_output_bias[phase_index]
        residual = (
            torch.einsum("btr,btor->bto", feature, selected_output)
            + selected_output_bias
        )
        pre_tanh_mean = self.action_head(recurrent) + residual
        mean = torch.tanh(pre_tanh_mean)
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        return RecurrentActorOutput(mean, pre_tanh_mean, log_std), next_state
