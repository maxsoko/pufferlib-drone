#!/usr/bin/env python3
"""Train the complete recurrent PufferNet policy on native DAgger trajectories.

The teacher may use privileged simulator state to label actions, but this model
only receives the same 32-value observation available to the deployed policy:
the original 23-value flight contract plus observable phase-gated adapter inputs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from policy_callable_checkpoint import CheckpointPolicy, _align
except ModuleNotFoundError:  # Imported as scripts.train_full_policy_bc.
    from scripts.policy_callable_checkpoint import CheckpointPolicy, _align


OBSERVATIONS = 32
ACTIONS = 4
RECORD_WIDTH = OBSERVATIONS + ACTIONS + 1
RACE_PHASE_INDICES = (23, 24, 25)
PREVIOUS_YAW_INDEX = 22


def _race_phase(
    observations: np.ndarray, *, gate_progress_observation: bool = False
) -> np.ndarray:
    if gate_progress_observation:
        return observations[:, RACE_PHASE_INDICES[0]]
    return (
        observations[:, RACE_PHASE_INDICES[0]] / 3.0
        + observations[:, RACE_PHASE_INDICES[1]] * (2.0 / 3.0)
        + observations[:, RACE_PHASE_INDICES[2]]
    )


def _mingru_g(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value >= 0.0, value + 0.5, torch.sigmoid(value))


def _mingru_log_g(value: torch.Tensor) -> torch.Tensor:
    return torch.where(
        value >= 0.0,
        torch.log(torch.relu(value) + 0.5),
        -torch.nn.functional.softplus(-value),
    )


class SequencePufferNet(nn.Module):
    def __init__(self, checkpoint: CheckpointPolicy, *, native_bf16: bool = False):
        super().__init__()
        self.hidden_dim = checkpoint.hidden_dim
        self.num_layers = checkpoint.num_layers
        self.native_bf16 = native_bf16
        self.encoder = nn.Parameter(torch.from_numpy(checkpoint.encoder.copy()))
        self.decoder = nn.Parameter(torch.from_numpy(checkpoint.decoder.copy()))
        self.mingru = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(weight.copy())) for weight in checkpoint.mingru_proj]
        )

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim,
            device=device,
            dtype=torch.bfloat16 if self.native_bf16 else self.encoder.dtype)

    def forward_chunk_outputs(
        self, observations: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # observations: [batch, time, obs]
        outputs: list[torch.Tensor] = []
        next_states = [state[layer] for layer in range(self.num_layers)]
        encoder = self.encoder
        decoder = self.decoder
        projections: list[torch.Tensor] = list(self.mingru)
        if self.native_bf16:
            encoder = encoder.to(torch.bfloat16)
            decoder = decoder.to(torch.bfloat16)
            projections = [projection.to(torch.bfloat16) for projection in projections]
        for time_index in range(observations.shape[1]):
            step_observation = observations[:, time_index]
            if self.native_bf16:
                step_observation = step_observation.to(torch.bfloat16)
            hidden = step_observation @ encoder.T
            updated_states: list[torch.Tensor] = []
            for layer_index, projection in enumerate(projections):
                projected = hidden @ projection.T
                candidate, gate, highway = projected.chunk(3, dim=-1)
                recurrent = next_states[layer_index] + torch.sigmoid(gate) * (
                    _mingru_g(candidate) - next_states[layer_index]
                )
                highway_gate = torch.sigmoid(highway)
                hidden = highway_gate * recurrent + (1.0 - highway_gate) * hidden
                updated_states.append(recurrent)
            next_states = updated_states
            outputs.append((hidden @ decoder.T).float())
        return torch.stack(outputs, dim=1), torch.stack(next_states, dim=0)

    def forward_chunk(
        self, observations: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, next_state = self.forward_chunk_outputs(observations, state)
        return outputs[..., :ACTIONS], next_state

    def encode_sequence_parallel(self, observations: torch.Tensor) -> torch.Tensor:
        """Return zero-state recurrent features using the parallel MinGRU scan."""
        encoder = self.encoder
        projections: list[torch.Tensor] = list(self.mingru)
        if self.native_bf16:
            observations = observations.to(torch.bfloat16)
            encoder = encoder.to(torch.bfloat16)
            projections = [projection.to(torch.bfloat16) for projection in projections]
        hidden = observations @ encoder.T
        for projection in projections:
            projected = hidden @ projection.T
            candidate, gate, highway = projected.chunk(3, dim=-1)
            log_coefficients = -torch.nn.functional.softplus(gate)
            log_values = -torch.nn.functional.softplus(-gate) + _mingru_log_g(candidate)
            cumulative = log_coefficients.cumsum(dim=1)
            recurrent = (
                cumulative
                + (log_values - cumulative).logcumsumexp(dim=1)
            ).exp()
            highway_gate = torch.sigmoid(highway)
            hidden = highway_gate * recurrent + (1.0 - highway_gate) * hidden
        return hidden

    def forward_sequence_parallel(self, observations: torch.Tensor) -> torch.Tensor:
        """Evaluate zero-state MinGRU sequences with the parallel scan used in training."""
        hidden = self.encode_sequence_parallel(observations)
        decoder = self.decoder.to(torch.bfloat16) if self.native_bf16 else self.decoder
        return (hidden @ decoder.T)[..., :ACTIONS].float()


def load_episodes(paths: list[Path]) -> list[np.ndarray]:
    episodes: list[np.ndarray] = []
    for path in paths:
        values = np.fromfile(path, dtype=np.float32)
        if values.size % RECORD_WIDTH:
            raise ValueError(f"{path} contains a partial teacher record")
        records = values.reshape(-1, RECORD_WIDTH)
        starts = np.flatnonzero(records[:, -1] > 0.5)
        if not len(starts) or starts[0] != 0:
            raise ValueError(f"{path} does not begin with an episode reset record")
        ends = np.r_[starts[1:], len(records)]
        episodes.extend(records[start:end] for start, end in zip(starts, ends))
    return episodes


def requires_gate_progress_observation(episodes: list[np.ndarray]) -> bool:
    """Detect the six-gate one-hot contract that legacy phase decoding corrupts."""
    return any(np.any(episode[:, 26:30] > 0.5) for episode in episodes)


def episode_batches(
    episodes: list[np.ndarray], batch_size: int, rng: random.Random
) -> list[list[np.ndarray]]:
    ordered = list(episodes)
    rng.shuffle(ordered)
    return [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]


def restore_unselected_encoder_features(
    encoder: torch.Tensor,
    source_encoder: torch.Tensor,
    selected_features: tuple[int, ...],
) -> None:
    """Undo optimizer-side drift outside explicitly selected input columns."""
    selected = torch.zeros(encoder.shape[1], dtype=torch.bool, device=encoder.device)
    selected[list(selected_features)] = True
    with torch.no_grad():
        encoder[:, ~selected] = source_encoder[:, ~selected]


def source_relative_targets(
    hidden: torch.Tensor,
    source_decoder: torch.Tensor,
    action_offset: torch.Tensor,
) -> torch.Tensor:
    """Return source-policy actions plus a bounded offline calibration delta."""
    source_actions = (hidden @ source_decoder.T)[..., :ACTIONS].float()
    return offset_source_actions(source_actions, action_offset)


def offset_source_actions(
    source_actions: torch.Tensor,
    action_offset: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a requested calibration delta over source-policy actions."""
    return source_actions + action_offset


def source_anchor_mask(
    mask: torch.Tensor,
    active: torch.Tensor,
    observations: torch.Tensor,
    race_phase: torch.Tensor,
    *,
    outside_active: bool,
    max_race_phase: float | None,
    until_elapsed_fraction: float,
) -> torch.Tensor:
    if outside_active:
        return mask & ~active
    if max_race_phase is not None:
        return mask & (race_phase < max_race_phase)
    return mask & (observations[..., 18] < until_elapsed_fraction)


def save_native_checkpoint(
    source: Path,
    output: Path,
    model: SequencePufferNet,
    *,
    layout_precision_bytes: int,
) -> None:
    serialized = np.fromfile(source, dtype=np.float32)
    tensors = [
        model.encoder.detach().cpu().numpy(),
        model.decoder.detach().cpu().numpy(),
        None,  # Preserve learned PPO log_std; deployment uses the mean action.
        *(weight.detach().cpu().numpy() for weight in model.mingru),
    ]
    counts = [
        model.encoder.numel(),
        model.decoder.numel(),
        ACTIONS,
        *[weight.numel() for weight in model.mingru],
    ]
    offset = 0
    for tensor, count in zip(tensors, counts):
        if tensor is not None and offset < len(serialized):
            flat = np.asarray(tensor, dtype=np.float32).reshape(-1)
            available = min(count, len(serialized) - offset)
            serialized[offset:offset + available] = flat[:available]
        offset = _align(offset + count, layout_precision_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized.tofile(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--chunk-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3385)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of MinGRU layers serialized in the native checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-layout-precision-bytes",
        type=int,
        choices=(2, 4),
        default=2,
        help="Native arena layout used by the input and output checkpoint.",
    )
    parser.add_argument(
        "--native-bf16", action="store_true",
        help="Quantize recurrent inference like the default CUDA backend.")
    parser.add_argument(
        "--freeze-recurrent", action="store_true",
        help="Train only the action decoder and preserve encoder/MinGRU weights.")
    parser.add_argument(
        "--train-last-recurrent-layer", action="store_true",
        help=(
            "Train the decoder and final MinGRU layer while preserving the encoder "
            "and earlier recurrent layers."
        ),
    )
    parser.add_argument(
        "--train-input-feature", type=int, action="append", default=None, metavar="INDEX",
        help=(
            "Train only the selected encoder input column; repeat for multiple "
            "features. Otherwise selected trainable layers are unchanged."
        ),
    )
    parser.add_argument(
        "--freeze-decoder", action="store_true",
        help="Preserve the complete decoder while training selected encoder/recurrent parameters.",
    )
    parser.add_argument(
        "--parallel-sequence", action="store_true",
        help="Use the zero-state parallel MinGRU scan for much faster full-episode BC.")
    parser.add_argument(
        "--restore-best-training-loss", action="store_true",
        help=(
            "Write the epoch with the lowest weighted active-course imitation loss "
            "instead of blindly writing the final epoch."
        ),
    )
    parser.add_argument(
        "--solve-decoder-ridge", type=float, default=None, metavar="LAMBDA",
        help=(
            "Freeze encoder/MinGRU and solve the linear action decoder exactly "
            "with nonnegative ridge regularization toward the source decoder "
            "instead of gradient epochs."
        ),
    )
    parser.add_argument(
        "--source-anchor-weight", type=float, default=0.0,
        help=(
            "Penalize action-decoder drift from the input checkpoint on early-course "
            "observations. With a trainable recurrent network this distills the full "
            "input policy; requires --parallel-sequence."
        ),
    )
    parser.add_argument(
        "--source-anchor-until-elapsed-fraction", type=float, default=0.23,
        help="Observation[18] cutoff for --source-anchor-weight (default: 0.23).",
    )
    parser.add_argument(
        "--source-anchor-max-race-phase",
        type=float,
        default=None,
        help=(
            "When set, anchor source-policy actions where the observable race "
            "phase is below this value instead of using elapsed time."
        ),
    )
    parser.add_argument(
        "--source-anchor-outside-active",
        action="store_true",
        help=(
            "Anchor the input policy on every record outside --min-race-phase/"
            "--max-race-phase. This is useful for a phase-local decoder correction."
        ),
    )
    parser.add_argument(
        "--source-relative-action-offset",
        type=float,
        nargs=4,
        default=None,
        metavar=("PITCH", "ROLL", "THRUST", "YAW"),
        help=(
            "Target the input policy action plus this per-action offset on the "
            "active race phase instead of using dataset labels. Requires the "
            "parallel sequence path."
        ),
    )
    parser.add_argument(
        "--restore-last-yaw-from-target",
        action="store_true",
        help=(
            "Replace observation[22] with the preceding teacher yaw before "
            "recurrent encoding (legacy datasets only)."
        ),
    )
    parser.add_argument(
        "--gate-progress-observation",
        action="store_true",
        help=(
            "Interpret observation[23] as normalized active_gate_index / num_gates "
            "instead of decoding the legacy one-hot phase adapter."
        ),
    )
    parser.add_argument(
        "--min-race-phase",
        type=float,
        default=0.0,
        help=(
            "Compute the imitation loss only within the appended one-hot race phase "
            "race phase is at least this value."
        ),
    )
    parser.add_argument(
        "--max-race-phase",
        type=float,
        default=1.0,
        help="Upper inclusive race-phase bound for the imitation loss.",
    )
    parser.add_argument(
        "--action-loss-weights",
        type=float,
        nargs=4,
        default=(1.0, 1.0, 1.0, 1.0),
        metavar=("PITCH", "ROLL", "THRUST", "YAW"),
        help=(
            "Per-action imitation weights for targeted course-phase corrections "
            "that preserve already-solved command channels."
        ),
    )
    parser.add_argument(
        "--preserve-source-action",
        type=int,
        action="append",
        default=[],
        metavar="INDEX",
        help=(
            "Use the input policy's action as the active-course target for this "
            "channel instead of the teacher target; repeat for multiple channels. "
            "Requires --parallel-sequence."
        ),
    )
    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.batch_episodes < 1
        or args.chunk_length < 1
        or args.num_layers < 1
    ):
        parser.error("epochs, batch episodes, chunk length, and num layers must be positive")
    if args.solve_decoder_ridge is not None and args.solve_decoder_ridge < 0.0:
        parser.error("--solve-decoder-ridge must be nonnegative")
    if args.solve_decoder_ridge is not None and not args.parallel_sequence:
        parser.error("--solve-decoder-ridge requires --parallel-sequence")
    if args.source_anchor_weight < 0.0:
        parser.error("--source-anchor-weight must be nonnegative")
    if not 0.0 <= args.source_anchor_until_elapsed_fraction <= 1.0:
        parser.error("--source-anchor-until-elapsed-fraction must be in [0, 1]")
    if args.source_anchor_max_race_phase is not None and not (
        0.0 <= args.source_anchor_max_race_phase <= 1.0
    ):
        parser.error("--source-anchor-max-race-phase must be in [0, 1]")
    if not 0.0 <= args.min_race_phase <= 1.0:
        parser.error("--min-race-phase must be in [0, 1]")
    if not 0.0 <= args.max_race_phase <= 1.0:
        parser.error("--max-race-phase must be in [0, 1]")
    if args.max_race_phase < args.min_race_phase:
        parser.error("--max-race-phase must be >= --min-race-phase")
    if any(weight < 0.0 for weight in args.action_loss_weights):
        parser.error("--action-loss-weights must be nonnegative")
    if not any(weight > 0.0 for weight in args.action_loss_weights):
        parser.error("at least one --action-loss-weights value must be positive")
    if any(not 0 <= action < ACTIONS for action in args.preserve_source_action):
        parser.error(f"--preserve-source-action must be in [0, {ACTIONS - 1}]")
    if len(set(args.preserve_source_action)) != len(args.preserve_source_action):
        parser.error("--preserve-source-action cannot repeat an action")
    if args.preserve_source_action and not args.parallel_sequence:
        parser.error("--preserve-source-action requires --parallel-sequence")
    if args.preserve_source_action and args.solve_decoder_ridge is not None:
        parser.error("--preserve-source-action is not supported with decoder ridge")
    if args.source_anchor_weight and not args.parallel_sequence:
        parser.error("--source-anchor-weight requires --parallel-sequence")
    if args.source_anchor_outside_active and not args.source_anchor_weight:
        parser.error("--source-anchor-outside-active requires --source-anchor-weight")
    if args.source_relative_action_offset is not None and not args.parallel_sequence:
        parser.error("--source-relative-action-offset requires --parallel-sequence")
    if args.solve_decoder_ridge is not None and not args.freeze_recurrent:
        parser.error("--solve-decoder-ridge requires --freeze-recurrent")
    if args.freeze_recurrent and args.train_last_recurrent_layer:
        parser.error("choose either --freeze-recurrent or --train-last-recurrent-layer")
    if args.train_input_feature is not None and any(
        not 0 <= feature < OBSERVATIONS for feature in args.train_input_feature
    ):
        parser.error(f"--train-input-feature must be in [0, {OBSERVATIONS - 1}]")
    torch.manual_seed(args.seed)
    randomizer = random.Random(args.seed)
    device = torch.device(args.device)
    action_loss_weights = torch.tensor(
        args.action_loss_weights, dtype=torch.float32, device=device)
    action_loss_weight_sum = float(sum(args.action_loss_weights))
    source_relative_action_offset = (
        torch.tensor(
            args.source_relative_action_offset, dtype=torch.float32, device=device)
        if args.source_relative_action_offset is not None
        else None
    )
    episodes = load_episodes(args.dataset)
    if requires_gate_progress_observation(episodes) and not args.gate_progress_observation:
        parser.error(
            "six-gate phase one-hot detected; --gate-progress-observation is required"
        )
    checkpoint = CheckpointPolicy.load(
        str(args.input),
        input_dim=OBSERVATIONS,
        num_layers=args.num_layers,
        layout_precision_bytes=args.checkpoint_layout_precision_bytes,
    )
    model = SequencePufferNet(checkpoint, native_bf16=args.native_bf16).to(device)
    source_decoder = model.decoder[:ACTIONS].detach().clone()
    if model.native_bf16:
        source_decoder = source_decoder.to(torch.bfloat16)
    source_model: SequencePufferNet | None = None
    if args.preserve_source_action or source_relative_action_offset is not None or (
        args.source_anchor_weight
        and (not args.freeze_recurrent or args.train_input_feature is not None)
    ):
        source_model = SequencePufferNet(checkpoint, native_bf16=args.native_bf16).to(device)
        source_model.eval()
        for parameter in source_model.parameters():
            parameter.requires_grad_(False)
    if args.freeze_recurrent or args.solve_decoder_ridge is not None:
        model.encoder.requires_grad_(False)
        for weight in model.mingru:
            weight.requires_grad_(False)
    elif args.train_last_recurrent_layer:
        model.encoder.requires_grad_(False)
        for weight in model.mingru[:-1]:
            weight.requires_grad_(False)
    if args.train_input_feature is not None:
        model.encoder.requires_grad_(True)
        feature_indices = tuple(args.train_input_feature)
        source_encoder = model.encoder.detach().clone()

        def preserve_other_encoder_features(gradient: torch.Tensor) -> torch.Tensor:
            masked = torch.zeros_like(gradient)
            masked[:, feature_indices] = gradient[:, feature_indices]
            return masked

        model.encoder.register_hook(preserve_other_encoder_features)
    if args.freeze_decoder:
        model.decoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay)

    total_records = sum(len(episode) for episode in episodes)
    print(
        f"loaded episodes={len(episodes)} records={total_records} device={device} "
        f"chunk_length={args.chunk_length} native_bf16={args.native_bf16} "
        f"freeze_recurrent={args.freeze_recurrent} "
        f"train_last_recurrent_layer={args.train_last_recurrent_layer} "
        f"train_input_feature={args.train_input_feature} "
        f"parallel_sequence={args.parallel_sequence} "
        f"restore_last_yaw_from_target={args.restore_last_yaw_from_target} "
        f"race_phase_range=[{args.min_race_phase}, {args.max_race_phase}] "
        f"action_loss_weights={tuple(args.action_loss_weights)} "
        f"preserve_source_action={tuple(args.preserve_source_action)}",
        flush=True,
    )
    if args.solve_decoder_ridge is not None:
        gram = torch.zeros(
            model.hidden_dim, model.hidden_dim, dtype=torch.float64)
        anchor_gram = torch.zeros_like(gram)
        cross = torch.zeros(model.hidden_dim, ACTIONS, dtype=torch.float64)
        target_square = torch.zeros(ACTIONS, dtype=torch.float64)
        samples = 0
        model.eval()
        with torch.no_grad():
            for batch in episode_batches(episodes, args.batch_episodes, randomizer):
                maximum_length = max(len(episode) for episode in batch)
                observations = torch.zeros(
                    len(batch), maximum_length, OBSERVATIONS,
                    dtype=torch.float32, device=device)
                targets = torch.zeros(
                    len(batch), maximum_length, ACTIONS,
                    dtype=torch.float32, device=device)
                mask = torch.zeros(
                    len(batch), maximum_length, dtype=torch.bool, device=device)
                race_phase = torch.zeros(
                    len(batch), maximum_length, dtype=torch.float32, device=device)
                for batch_index, episode in enumerate(batch):
                    length = len(episode)
                    episode_observations = episode[:, :OBSERVATIONS].copy()
                    race_phase[batch_index, :length] = torch.from_numpy(
                        _race_phase(
                            episode_observations,
                            gate_progress_observation=args.gate_progress_observation,
                        ).copy()).to(device)
                    if args.restore_last_yaw_from_target:
                        episode_observations[0, PREVIOUS_YAW_INDEX] = 0.0
                        episode_observations[1:, PREVIOUS_YAW_INDEX] = episode[
                            :-1, OBSERVATIONS + 3]
                    observations[batch_index, :length] = torch.from_numpy(
                        episode_observations).to(device)
                    targets[batch_index, :length] = torch.from_numpy(
                        episode[:, OBSERVATIONS:OBSERVATIONS + ACTIONS].copy()).to(device)
                    mask[batch_index, :length] = True
                hidden = model.encode_sequence_parallel(observations)
                active = mask & (race_phase >= args.min_race_phase) & (
                    race_phase <= args.max_race_phase
                )
                active_hidden = hidden[active]
                active_targets = targets[active]
                if source_relative_action_offset is not None:
                    active_targets = source_relative_targets(
                        active_hidden,
                        source_decoder,
                        source_relative_action_offset,
                    )
                active_hidden_float = active_hidden.float()
                gram += (
                    active_hidden_float.T @ active_hidden_float
                ).double().cpu()
                if args.source_anchor_weight:
                    anchor_mask = source_anchor_mask(
                        mask,
                        active,
                        observations,
                        race_phase,
                        outside_active=args.source_anchor_outside_active,
                        max_race_phase=args.source_anchor_max_race_phase,
                        until_elapsed_fraction=(
                            args.source_anchor_until_elapsed_fraction
                        ),
                    )
                    anchor_hidden = hidden[anchor_mask]
                    anchor_hidden_float = anchor_hidden.float()
                    anchor_gram += (
                        anchor_hidden_float.T @ anchor_hidden_float
                    ).double().cpu()
                cross += (
                    active_hidden_float.T @ active_targets.float()
                ).double().cpu()
                target_square += torch.sum(
                    active_targets.double() ** 2, dim=0).cpu()
                samples += active_hidden.shape[0]

        regularization = float(args.solve_decoder_ridge)
        source_weights = model.decoder[:ACTIONS].detach().double().cpu()
        anchor_weight = float(args.source_anchor_weight)
        system = gram + anchor_weight * anchor_gram + regularization * torch.eye(
            model.hidden_dim, dtype=torch.float64)
        solution = torch.linalg.solve(
            system,
            cross
            + anchor_weight * (anchor_gram @ source_weights.T)
            + regularization * source_weights.T,
        )
        weights = solution.T
        for action, loss_weight in enumerate(args.action_loss_weights):
            if loss_weight == 0.0:
                weights[action].copy_(source_weights[action])
        with torch.no_grad():
            model.decoder[:ACTIONS].copy_(
                weights.to(device=device, dtype=model.decoder.dtype))
        squared_error = target_square - 2.0 * torch.sum(
            weights * cross.T, dim=1)
        squared_error += torch.einsum("ah,hk,ak->a", weights, gram, weights)
        rmse = torch.sqrt(torch.clamp(squared_error / max(samples, 1), min=0.0))
        print(
            f"decoder_ridge={regularization:.9g} samples={samples} "
            f"source_anchor_weight={anchor_weight:.9g} "
            f"mean_gram_diagonal={torch.diagonal(gram).mean():.6f} "
            f"rmse=({rmse[0]:.6f},{rmse[1]:.6f},{rmse[2]:.6f},{rmse[3]:.6f})",
            flush=True,
        )
        save_native_checkpoint(
            args.input,
            args.output,
            model,
            layout_precision_bytes=args.checkpoint_layout_precision_bytes,
        )
        CheckpointPolicy.load(
            str(args.output),
            input_dim=OBSERVATIONS,
            num_layers=args.num_layers,
            layout_precision_bytes=args.checkpoint_layout_precision_bytes,
        )
        print(f"wrote {args.output}", flush=True)
        return 0

    best_training_loss = float("inf")
    best_training_epoch = 0
    best_training_state = None
    for epoch in range(1, args.epochs + 1):
        squared_error = torch.zeros(ACTIONS, dtype=torch.float64, device=device)
        samples = 0
        for batch in episode_batches(episodes, args.batch_episodes, randomizer):
            maximum_length = max(len(episode) for episode in batch)
            batch_count = len(batch)
            observations = torch.zeros(
                batch_count, maximum_length, OBSERVATIONS, dtype=torch.float32, device=device)
            targets = torch.zeros(
                batch_count, maximum_length, ACTIONS, dtype=torch.float32, device=device)
            mask = torch.zeros(
                batch_count, maximum_length, dtype=torch.bool, device=device)
            race_phase = torch.zeros(
                batch_count, maximum_length, dtype=torch.float32, device=device)
            for batch_index, episode in enumerate(batch):
                length = len(episode)
                episode_observations = episode[:, :OBSERVATIONS].copy()
                race_phase[batch_index, :length] = torch.from_numpy(
                    _race_phase(
                        episode_observations,
                        gate_progress_observation=args.gate_progress_observation,
                    ).copy()).to(device)
                if args.restore_last_yaw_from_target:
                    episode_observations[0, PREVIOUS_YAW_INDEX] = 0.0
                    episode_observations[1:, PREVIOUS_YAW_INDEX] = episode[
                        :-1, OBSERVATIONS + 3]
                observations[batch_index, :length] = torch.from_numpy(
                    episode_observations).to(device)
                targets[batch_index, :length] = torch.from_numpy(
                    episode[:, OBSERVATIONS:OBSERVATIONS + ACTIONS].copy()).to(device)
                mask[batch_index, :length] = True

            if args.parallel_sequence:
                hidden = model.encode_sequence_parallel(observations)
                decoder = (
                    model.decoder.to(torch.bfloat16)
                    if model.native_bf16 else model.decoder
                )
                prediction = (hidden @ decoder.T)[..., :ACTIONS].float()
                effective_targets = targets
                source_prediction = None
                if source_relative_action_offset is not None:
                    if source_model is None:
                        raise RuntimeError(
                            "source-relative target model was not initialized"
                        )
                    with torch.no_grad():
                        source_prediction = source_model.forward_sequence_parallel(
                            observations
                        )
                    effective_targets = offset_source_actions(
                        source_prediction,
                        source_relative_action_offset,
                    )
                if args.preserve_source_action:
                    if source_model is None:
                        raise RuntimeError("source action preservation model was not initialized")
                    if source_prediction is None:
                        with torch.no_grad():
                            source_prediction = source_model.forward_sequence_parallel(
                                observations
                            )
                    effective_targets = effective_targets.clone()
                    effective_targets[..., args.preserve_source_action] = (
                        source_prediction[..., args.preserve_source_action]
                    )
                errors = prediction - effective_targets
                active = mask & (race_phase >= args.min_race_phase) & (
                    race_phase <= args.max_race_phase
                )
                active_errors = errors[active]
                loss_numerator = torch.sum(
                    active_errors * active_errors * action_loss_weights)
                loss_denominator = active_errors.shape[0] * action_loss_weight_sum
                if args.source_anchor_weight:
                    anchor_mask = source_anchor_mask(
                        mask,
                        active,
                        observations,
                        race_phase,
                        outside_active=args.source_anchor_outside_active,
                        max_race_phase=args.source_anchor_max_race_phase,
                        until_elapsed_fraction=(
                            args.source_anchor_until_elapsed_fraction
                        ),
                    )
                    if source_model is None:
                        source_prediction = hidden @ source_decoder.T
                    else:
                        with torch.no_grad():
                            source_prediction = source_model.forward_sequence_parallel(
                                observations
                            )
                    anchor_errors = prediction[anchor_mask] - source_prediction[anchor_mask]
                    loss_numerator = loss_numerator + args.source_anchor_weight * torch.sum(
                        anchor_errors * anchor_errors * action_loss_weights
                    )
                    loss_denominator += (
                        args.source_anchor_weight
                        * anchor_errors.shape[0]
                        * action_loss_weight_sum
                    )
                loss = loss_numerator / loss_denominator
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.train_input_feature is not None:
                    restore_unselected_encoder_features(
                        model.encoder, source_encoder, feature_indices)
                squared_error += torch.sum(
                    active_errors.detach().double() ** 2, dim=0)
                samples += active_errors.shape[0]
                continue

            state = model.initial_state(batch_count, device)
            for start in range(0, maximum_length, args.chunk_length):
                end = min(start + args.chunk_length, maximum_length)
                active = mask[:, start:end] & (
                    race_phase[:, start:end] >= args.min_race_phase
                ) & (
                    race_phase[:, start:end] <= args.max_race_phase
                )
                if not active.any():
                    continue
                prediction, state = model.forward_chunk(observations[:, start:end], state)
                errors = prediction - targets[:, start:end]
                active_errors = errors[active]
                loss = torch.sum(
                    active_errors * active_errors * action_loss_weights
                ) / (active_errors.shape[0] * action_loss_weight_sum)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.train_input_feature is not None:
                    restore_unselected_encoder_features(
                        model.encoder, source_encoder, feature_indices)
                squared_error += torch.sum(
                    active_errors.detach().double() ** 2, dim=0)
                samples += active_errors.shape[0]
                state = state.detach()

        rmse = torch.sqrt(squared_error / max(samples, 1)).cpu().numpy()
        print(
            f"epoch={epoch}/{args.epochs} samples={samples} "
            f"rmse=({rmse[0]:.6f},{rmse[1]:.6f},{rmse[2]:.6f},{rmse[3]:.6f})",
            flush=True,
        )
        if args.restore_best_training_loss:
            weighted_loss = float(
                torch.sum(squared_error * action_loss_weights.double()).item()
                / max(samples * action_loss_weight_sum, 1.0)
            )
            if weighted_loss < best_training_loss:
                best_training_loss = weighted_loss
                best_training_epoch = epoch
                best_training_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

    if best_training_state is not None:
        model.load_state_dict(best_training_state)
        print(
            f"restored_best_epoch={best_training_epoch} "
            f"weighted_active_mse={best_training_loss:.9f}",
            flush=True,
        )

    save_native_checkpoint(
        args.input,
        args.output,
        model,
        layout_precision_bytes=args.checkpoint_layout_precision_bytes,
    )
    CheckpointPolicy.load(
        str(args.output),
        input_dim=OBSERVATIONS,
        num_layers=args.num_layers,
        layout_precision_bytes=args.checkpoint_layout_precision_bytes,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
