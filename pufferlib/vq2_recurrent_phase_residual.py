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
from pufferlib.vq2_public_phase import (
    LONG_COURSE_GATE_CAP,
    OFFICIAL_PROGRESS_SCALE,
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


class VQ2UnboundedProgressMLPResidualActor(VQ2PhaseRecurrentActor):
    """Long-course actor with unsaturated index/6 progress and sparse heads.

    Residual heads are tracked through the 32-gate native curriculum. Beyond
    that range the continuous recurrent base remains active and the residual
    is exact zero. The official finish condition is never encoded here.
    """

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
        phases = LONG_COURSE_GATE_CAP + 1
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

    def load_converted_state(
        self, legacy_state: dict[str, torch.Tensor]
    ) -> None:
        """Convert one index/16 17-head actor without changing its actions."""

        residual_names = {
            "indexed_phase_residual_input",
            "indexed_phase_residual_input_bias",
            "indexed_phase_residual_output",
            "indexed_phase_residual_output_bias",
        }
        expected = set(self.state_dict())
        if set(legacy_state) != expected:
            # Shapes differ but names must remain exactly the same.
            raise RuntimeError("legacy actor keys changed")
        current = self.state_dict()
        for name in expected - residual_names:
            value = legacy_state[name].detach().to(
                device=current[name].device, dtype=current[name].dtype
            )
            if value.shape != current[name].shape:
                raise RuntimeError(f"legacy base parameter shape changed: {name}")
            current[name].copy_(value)
        # Preserve W*(index/16) exactly under the new index/6 input scale.
        current["phase_embedding.weight"].copy_(
            legacy_state["phase_embedding.weight"].detach().to(
                current["phase_embedding.weight"]
            ) * (OFFICIAL_PROGRESS_SCALE / OFFICIAL_GATE_COUNT)
        )
        legacy_phases = OFFICIAL_GATE_COUNT + 1
        for name in residual_names:
            source = legacy_state[name].detach().to(
                device=current[name].device, dtype=current[name].dtype
            )
            if source.shape[0] != legacy_phases or source.shape[1:] != current[name].shape[1:]:
                raise RuntimeError(f"legacy residual parameter shape changed: {name}")
            if "output" in name:
                current[name].zero_()
            current[name][:legacy_phases].copy_(source)
        self.load_state_dict(current)

    def forward_sequence(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 3 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("unbounded-progress actor accepts [batch,time,4119] only")
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        if tuple(state.shape) != (1, batch, self.hidden_size):
            raise ValueError("unbounded-progress recurrent state shape changed")
        legal = observation[..., :LEGAL_OBS_SIZE]
        progress = observation[..., LEGAL_OBS_SIZE:]
        if not bool(torch.isfinite(progress).all()) or bool((progress < -1e-6).any()):
            raise ValueError("public gate progress must be finite and nonnegative")
        encoded = self.encoder(legal) + self.phase_embedding(progress)
        recurrent, next_state = self.recurrent(encoded, state)
        raw_index = torch.round(progress[..., 0] * OFFICIAL_PROGRESS_SCALE)
        tracked = raw_index <= LONG_COURSE_GATE_CAP
        phase_index = raw_index.to(torch.long).clamp_(0, LONG_COURSE_GATE_CAP)
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
        ) * tracked[..., None]
        pre_tanh_mean = self.action_head(recurrent) + residual
        mean = torch.tanh(pre_tanh_mean)
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        return RecurrentActorOutput(mean, pre_tanh_mean, log_std), next_state


class VQ2PhaseLocalAdapterActor(VQ2UnboundedProgressMLPResidualActor):
    """Base long-course actor plus a recurrent adapter at one public phase.

    The adapter consumes only the base policy's legal-observation recurrent
    state. Its state remains zero before the configured phase, its output is
    gated off everywhere else, and a zero output head is exactly base-equivalent.
    """

    def __init__(
        self,
        *,
        target_phase: int,
        hidden_size: int = 256,
        residual_size: int = 64,
        adapter_size: int = 64,
        initial_std: float = 0.15,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            residual_size=residual_size,
            initial_std=initial_std,
        )
        if not 0 <= target_phase <= LONG_COURSE_GATE_CAP:
            raise ValueError("adapter target phase is outside the tracked course")
        if adapter_size <= 0:
            raise ValueError("adapter size must be positive")
        self.target_phase = int(target_phase)
        self.adapter_size = int(adapter_size)
        self.phase_adapter_cell = nn.GRUCell(hidden_size, adapter_size)
        self.phase_adapter_output = nn.Linear(adapter_size, ACTION_SIZE)
        nn.init.zeros_(self.phase_adapter_output.weight)
        nn.init.zeros_(self.phase_adapter_output.bias)

    def load_base_state(self, base_state: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(base_state, strict=False)
        expected = {
            "phase_adapter_cell.weight_ih",
            "phase_adapter_cell.weight_hh",
            "phase_adapter_cell.bias_ih",
            "phase_adapter_cell.bias_hh",
            "phase_adapter_output.weight",
            "phase_adapter_output.bias",
        }
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected:
            raise RuntimeError("adapter base differs outside phase-local state")
        nn.init.zeros_(self.phase_adapter_output.weight)
        nn.init.zeros_(self.phase_adapter_output.bias)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        return torch.zeros(
            1,
            batch_size,
            self.hidden_size + self.adapter_size,
            device=device,
            dtype=dtype,
        )

    def forward_step(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("phase-local adapter step accepts [batch,4119] only")
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        expected = (1, batch, self.hidden_size + self.adapter_size)
        if tuple(state.shape) != expected:
            raise ValueError("phase-local adapter recurrent state shape changed")
        base_state = state[..., : self.hidden_size].contiguous()
        adapter_state = state[0, :, self.hidden_size :].contiguous()
        base_output, next_base_state = (
            VQ2UnboundedProgressMLPResidualActor.forward_sequence(
                self, observation.unsqueeze(1), base_state
            )
        )
        progress = observation[:, LEGAL_OBS_SIZE]
        raw_index = torch.round(progress * OFFICIAL_PROGRESS_SCALE).to(torch.long)
        active = raw_index == self.target_phase
        candidate_adapter = self.phase_adapter_cell(
            next_base_state[0], adapter_state
        )
        next_adapter = torch.where(
            active[:, None], candidate_adapter, adapter_state
        )
        adapter_residual = self.phase_adapter_output(next_adapter)
        adapter_residual = adapter_residual * active[:, None]
        pre_tanh_mean = base_output.pre_tanh_mean[:, 0] + adapter_residual
        mean = torch.tanh(pre_tanh_mean)
        log_std = base_output.log_std[:, 0]
        next_state = torch.cat((next_base_state, next_adapter[None]), dim=-1)
        return RecurrentActorOutput(mean, pre_tanh_mean, log_std), next_state

    def forward_sequence(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 3 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("phase-local adapter accepts [batch,time,4119] only")
        batch, steps, _ = observation.shape
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        means: list[torch.Tensor] = []
        pre_tanh_means: list[torch.Tensor] = []
        log_stds: list[torch.Tensor] = []
        next_state = state
        for step in range(steps):
            output, next_state = self.forward_step(observation[:, step], next_state)
            means.append(output.mean)
            pre_tanh_means.append(output.pre_tanh_mean)
            log_stds.append(output.log_std)
        return (
            RecurrentActorOutput(
                torch.stack(means, dim=1),
                torch.stack(pre_tanh_means, dim=1),
                torch.stack(log_stds, dim=1),
            ),
            next_state,
        )


class VQ2StackedPhaseRangeAdapterActor(VQ2PhaseLocalAdapterActor):
    """A frozen phase-local actor plus one later phase-range adapter.

    This preserves the first adapter's weights, output, and recurrent state
    exactly. The continuation adapter consumes the same public-observation
    base recurrent output, updates only inside ``[phase_min, phase_max)``, and
    emits the complete four-action residual only in that range.
    """

    def __init__(
        self,
        *,
        target_phase: int,
        continuation_phase_min: int,
        continuation_phase_max_exclusive: int,
        hidden_size: int = 256,
        residual_size: int = 64,
        adapter_size: int = 64,
        continuation_adapter_size: int = 64,
        initial_std: float = 0.15,
    ) -> None:
        super().__init__(
            target_phase=target_phase,
            hidden_size=hidden_size,
            residual_size=residual_size,
            adapter_size=adapter_size,
            initial_std=initial_std,
        )
        if not 0 <= continuation_phase_min < continuation_phase_max_exclusive:
            raise ValueError("continuation phase range is invalid")
        if continuation_phase_max_exclusive > LONG_COURSE_GATE_CAP + 1:
            raise ValueError("continuation phase range exceeds tracked course")
        if continuation_adapter_size <= 0:
            raise ValueError("continuation adapter size must be positive")
        self.continuation_phase_min = int(continuation_phase_min)
        self.continuation_phase_max_exclusive = int(
            continuation_phase_max_exclusive
        )
        self.continuation_adapter_size = int(continuation_adapter_size)
        self.continuation_adapter_cell = nn.GRUCell(
            hidden_size, continuation_adapter_size
        )
        self.continuation_adapter_output = nn.Linear(
            continuation_adapter_size, ACTION_SIZE
        )
        nn.init.zeros_(self.continuation_adapter_output.weight)
        nn.init.zeros_(self.continuation_adapter_output.bias)

    def load_phase_local_state(
        self, phase_local_state: dict[str, torch.Tensor]
    ) -> None:
        incompatible = self.load_state_dict(phase_local_state, strict=False)
        expected = {
            "continuation_adapter_cell.weight_ih",
            "continuation_adapter_cell.weight_hh",
            "continuation_adapter_cell.bias_ih",
            "continuation_adapter_cell.bias_hh",
            "continuation_adapter_output.weight",
            "continuation_adapter_output.bias",
        }
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected:
            raise RuntimeError("stacked adapter parent differs outside continuation state")
        nn.init.zeros_(self.continuation_adapter_output.weight)
        nn.init.zeros_(self.continuation_adapter_output.bias)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        return torch.zeros(
            1,
            batch_size,
            self.hidden_size + self.adapter_size + self.continuation_adapter_size,
            device=device,
            dtype=dtype,
        )

    def forward_step(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("stacked phase adapter step accepts [batch,4119] only")
        batch = observation.shape[0]
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        expected = (
            1,
            batch,
            self.hidden_size + self.adapter_size + self.continuation_adapter_size,
        )
        if tuple(state.shape) != expected:
            raise ValueError("stacked phase adapter recurrent state shape changed")
        phase15_end = self.hidden_size + self.adapter_size
        base_state = state[..., : self.hidden_size].contiguous()
        phase_adapter_state = state[0, :, self.hidden_size : phase15_end].contiguous()
        continuation_state = state[0, :, phase15_end:].contiguous()
        base_output, next_base_state = (
            VQ2UnboundedProgressMLPResidualActor.forward_sequence(
                self, observation.unsqueeze(1), base_state
            )
        )
        progress = observation[:, LEGAL_OBS_SIZE]
        raw_index = torch.round(progress * OFFICIAL_PROGRESS_SCALE).to(torch.long)

        phase_active = raw_index == self.target_phase
        candidate_phase_state = self.phase_adapter_cell(
            next_base_state[0], phase_adapter_state
        )
        next_phase_state = torch.where(
            phase_active[:, None], candidate_phase_state, phase_adapter_state
        )
        phase_residual = self.phase_adapter_output(next_phase_state)
        phase_residual = phase_residual * phase_active[:, None]

        continuation_active = (raw_index >= self.continuation_phase_min) & (
            raw_index < self.continuation_phase_max_exclusive
        )
        candidate_continuation_state = self.continuation_adapter_cell(
            next_base_state[0], continuation_state
        )
        next_continuation_state = torch.where(
            continuation_active[:, None],
            candidate_continuation_state,
            continuation_state,
        )
        continuation_residual = self.continuation_adapter_output(
            next_continuation_state
        )
        continuation_residual = continuation_residual * continuation_active[:, None]

        pre_tanh_mean = (
            base_output.pre_tanh_mean[:, 0]
            + phase_residual
            + continuation_residual
        )
        mean = torch.tanh(pre_tanh_mean)
        next_state = torch.cat(
            (
                next_base_state,
                next_phase_state[None],
                next_continuation_state[None],
            ),
            dim=-1,
        )
        return RecurrentActorOutput(mean, pre_tanh_mean, base_output.log_std[:, 0]), next_state

    def forward_sequence(
        self,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[RecurrentActorOutput, torch.Tensor]:
        if observation.ndim != 3 or observation.shape[-1] != PHASE_LEGAL_OBS_SIZE:
            raise ValueError("stacked phase adapter accepts [batch,time,4119] only")
        batch, steps, _ = observation.shape
        if state is None:
            state = self.initial_state(
                batch, device=observation.device, dtype=observation.dtype
            )
        means: list[torch.Tensor] = []
        pre_tanh_means: list[torch.Tensor] = []
        log_stds: list[torch.Tensor] = []
        next_state = state
        for step in range(steps):
            output, next_state = self.forward_step(observation[:, step], next_state)
            means.append(output.mean)
            pre_tanh_means.append(output.pre_tanh_mean)
            log_stds.append(output.log_std)
        return (
            RecurrentActorOutput(
                torch.stack(means, dim=1),
                torch.stack(pre_tanh_means, dim=1),
                torch.stack(log_stds, dim=1),
            ),
            next_state,
        )
