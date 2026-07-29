#!/usr/bin/env python3
"""Fit a late-course decoder residual in the early-policy feature nullspace.

The recurrent policy and all non-action weights remain byte-identical to the
base checkpoint.  Early hidden-feature covariance defines a dormant subspace;
least squares then fits teacher residuals after the cutoff only in that
subspace.  This creates a single policy whose added behavior activates late,
without a runtime stage switch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from policy_callable_checkpoint import CheckpointPolicy, _align8
from train_full_policy_bc import (
    ACTIONS,
    OBSERVATIONS,
    SequencePufferNet,
    episode_batches,
    load_episodes,
)


def _batch_tensors(
    batch: list[np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum_length = max(len(episode) for episode in batch)
    observations = torch.zeros(
        len(batch), maximum_length, OBSERVATIONS, dtype=torch.float32, device=device
    )
    targets = torch.zeros(
        len(batch), maximum_length, ACTIONS, dtype=torch.float32, device=device
    )
    valid = torch.zeros(
        len(batch), maximum_length, dtype=torch.bool, device=device
    )
    for batch_index, episode in enumerate(batch):
        length = len(episode)
        observations[batch_index, :length] = torch.from_numpy(
            episode[:, :OBSERVATIONS].copy()
        ).to(device)
        targets[batch_index, :length] = torch.from_numpy(
            episode[:, OBSERVATIONS:OBSERVATIONS + ACTIONS].copy()
        ).to(device)
        valid[batch_index, :length] = True
    return observations, targets, valid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--elapsed-cutoff", type=float, default=0.23)
    parser.add_argument(
        "--late-race-phase",
        type=float,
        help=(
            "When set, define early/late records from the observable race phase "
            "instead of elapsed fraction. Gate 3 in the 32-input adapter is 1.0."
        ),
    )
    parser.add_argument("--max-early-eigenvalue", type=float, default=1e-8)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--batch-episodes", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input-dim", type=int, default=OBSERVATIONS)
    parser.add_argument(
        "--native-bf16",
        action="store_true",
        help="Compute recurrent features with the CUDA deployment precision.",
    )
    parser.add_argument(
        "--action-loss-weights",
        type=float,
        nargs=4,
        default=(1.0, 1.0, 1.0, 1.0),
        metavar=("PITCH", "ROLL", "THRUST", "YAW"),
        help="Zero preserves the corresponding source decoder row exactly.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.elapsed_cutoff <= 1.0:
        parser.error("--elapsed-cutoff must be in [0, 1]")
    if args.late_race_phase is not None and not 0.0 <= args.late_race_phase <= 1.0:
        parser.error("--late-race-phase must be in [0, 1]")
    if args.max_early_eigenvalue < 0.0 or args.ridge < 0.0:
        parser.error("eigenvalue threshold and ridge must be nonnegative")
    if args.input_dim != OBSERVATIONS:
        parser.error(f"this dataset contract requires --input-dim {OBSERVATIONS}")
    if any(weight < 0.0 for weight in args.action_loss_weights):
        parser.error("--action-loss-weights must be nonnegative")
    if not any(weight > 0.0 for weight in args.action_loss_weights):
        parser.error("at least one action loss weight must be positive")

    checkpoint = CheckpointPolicy.load(str(args.base), input_dim=args.input_dim)
    device = torch.device(args.device)
    model = SequencePufferNet(checkpoint, native_bf16=args.native_bf16).to(device).eval()
    episodes = load_episodes(args.dataset)
    batches = episode_batches(episodes, args.batch_episodes, random.Random(0))

    early_covariance_sum = torch.zeros(
        model.hidden_dim, model.hidden_dim, dtype=torch.float64
    )
    early_samples = 0
    with torch.no_grad():
        for batch in batches:
            observations, _targets, valid = _batch_tensors(batch, device)
            if args.late_race_phase is None:
                early = valid & (observations[..., 18] < args.elapsed_cutoff)
            else:
                phases = (
                    observations[..., 23] / 3.0
                    + observations[..., 24] * (2.0 / 3.0)
                    + observations[..., 25]
                )
                early = valid & (phases < args.late_race_phase)
            hidden = model.encode_sequence_parallel(observations)[early].double()
            early_covariance_sum += (hidden.T @ hidden).cpu()
            early_samples += hidden.shape[0]
    if early_samples == 0:
        raise ValueError("no records matched the early-course cutoff")

    early_covariance = early_covariance_sum / early_samples
    eigenvalues, eigenvectors = torch.linalg.eigh(early_covariance)
    selected = eigenvalues <= args.max_early_eigenvalue
    if not torch.any(selected):
        raise ValueError("eigenvalue threshold retained no dormant feature directions")
    basis = eigenvectors[:, selected]
    dimensions = basis.shape[1]

    gram = torch.zeros(dimensions, dimensions, dtype=torch.float64)
    cross = torch.zeros(dimensions, ACTIONS, dtype=torch.float64)
    target_square = torch.zeros(ACTIONS, dtype=torch.float64)
    late_samples = 0
    base_decoder = torch.from_numpy(checkpoint.decoder[:ACTIONS]).double()
    with torch.no_grad():
        for batch in batches:
            observations, targets, valid = _batch_tensors(batch, device)
            if args.late_race_phase is None:
                late = valid & (observations[..., 18] >= args.elapsed_cutoff)
            else:
                phases = (
                    observations[..., 23] / 3.0
                    + observations[..., 24] * (2.0 / 3.0)
                    + observations[..., 25]
                )
                late = valid & (phases >= args.late_race_phase)
            hidden = model.encode_sequence_parallel(observations)[late].double().cpu()
            active_targets = targets[late].double().cpu()
            residual_targets = active_targets - hidden @ base_decoder.T
            dormant_features = hidden @ basis
            gram += dormant_features.T @ dormant_features
            cross += dormant_features.T @ residual_targets
            target_square += torch.sum(residual_targets * residual_targets, dim=0)
            late_samples += hidden.shape[0]
    if late_samples == 0:
        raise ValueError("no records matched the late-course cutoff")

    gram /= late_samples
    cross /= late_samples
    target_square /= late_samples
    coefficients = torch.linalg.solve(
        gram + args.ridge * torch.eye(dimensions, dtype=torch.float64), cross
    )
    decoder_delta = coefficients.T @ basis.T
    for action, loss_weight in enumerate(args.action_loss_weights):
        if loss_weight == 0.0:
            decoder_delta[action].zero_()
    squared_error = target_square - 2.0 * torch.sum(coefficients.T * cross.T, dim=1)
    squared_error += torch.einsum(
        "ak,kl,al->a", coefficients.T, gram, coefficients.T
    )
    fit_rmse = torch.sqrt(torch.clamp(squared_error, min=0.0))
    early_rms = torch.sqrt(torch.clamp(
        torch.einsum("ah,hk,ak->a", decoder_delta, early_covariance, decoder_delta),
        min=0.0,
    ))

    serialized = np.fromfile(args.base, dtype=np.float32)
    decoder_offset = _align8(checkpoint.hidden_dim * checkpoint.input_dim)
    action_count = ACTIONS * checkpoint.hidden_dim
    serialized[decoder_offset:decoder_offset + action_count] = (
        checkpoint.decoder[:ACTIONS] + decoder_delta.float().numpy()
    ).reshape(-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized.tofile(args.output)
    CheckpointPolicy.load(str(args.output), input_dim=args.input_dim)

    print(
        f"wrote={args.output} early_samples={early_samples} late_samples={late_samples} "
        f"dormant_dimensions={dimensions}/{model.hidden_dim} "
        f"threshold={args.max_early_eigenvalue:.9g} ridge={args.ridge:.9g} "
        f"decoder_delta_l2={torch.linalg.norm(decoder_delta):.9f} "
        f"early_rms=({','.join(f'{value:.9g}' for value in early_rms.tolist())}) "
        f"late_fit_rmse=({','.join(f'{value:.9g}' for value in fit_rmse.tolist())})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
