#!/usr/bin/env python3
"""Project a late-course decoder update out of early-policy feature space.

The base recurrent policy is never changed.  A successful candidate's action
decoder residual is retained only in hidden-feature directions whose early
course covariance is below a chosen eigenvalue.  The resulting checkpoint is
still one ordinary recurrent PufferNet policy with no runtime gate or arbiter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from policy_callable_checkpoint import CheckpointPolicy, _align8
from train_full_policy_bc import ACTIONS, OBSERVATIONS, SequencePufferNet, episode_batches, load_episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--elapsed-cutoff", type=float, default=0.23)
    parser.add_argument("--max-early-eigenvalue", type=float, required=True)
    parser.add_argument("--batch-episodes", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not 0.0 <= args.elapsed_cutoff <= 1.0:
        parser.error("--elapsed-cutoff must be in [0, 1]")
    if args.max_early_eigenvalue < 0.0:
        parser.error("--max-early-eigenvalue must be nonnegative")

    base_checkpoint = CheckpointPolicy.load(str(args.base))
    candidate_checkpoint = CheckpointPolicy.load(str(args.candidate))
    if not np.array_equal(base_checkpoint.encoder, candidate_checkpoint.encoder):
        raise ValueError("candidate encoder differs from base; expected decoder-only candidate")
    for index, (base_weight, candidate_weight) in enumerate(
        zip(base_checkpoint.mingru_proj, candidate_checkpoint.mingru_proj)
    ):
        if not np.array_equal(base_weight, candidate_weight):
            raise ValueError(
                f"candidate MinGRU layer {index} differs from base; expected decoder-only candidate"
            )

    device = torch.device(args.device)
    model = SequencePufferNet(base_checkpoint).to(device).eval()
    episodes = load_episodes(args.dataset)
    covariance_sum = torch.zeros(
        model.hidden_dim, model.hidden_dim, dtype=torch.float64
    )
    samples = 0
    generator = random.Random(0)
    with torch.no_grad():
        for batch in episode_batches(episodes, args.batch_episodes, generator):
            maximum_length = max(len(episode) for episode in batch)
            observations = torch.zeros(
                len(batch), maximum_length, OBSERVATIONS,
                dtype=torch.float32, device=device,
            )
            valid = torch.zeros(
                len(batch), maximum_length, dtype=torch.bool, device=device
            )
            for batch_index, episode in enumerate(batch):
                length = len(episode)
                observations[batch_index, :length] = torch.from_numpy(
                    episode[:, :OBSERVATIONS].copy()
                ).to(device)
                valid[batch_index, :length] = True
            early = valid & (observations[..., 18] < args.elapsed_cutoff)
            hidden = model.encode_sequence_parallel(observations)[early].double()
            covariance_sum += (hidden.T @ hidden).cpu()
            samples += hidden.shape[0]

    if samples == 0:
        raise ValueError("no early-course records matched the elapsed cutoff")
    covariance = covariance_sum / samples
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    selected = eigenvalues <= args.max_early_eigenvalue
    if not torch.any(selected):
        raise ValueError("eigenvalue threshold retained no hidden-feature directions")
    basis = eigenvectors[:, selected]
    projector = basis @ basis.T

    source_delta = torch.from_numpy(
        candidate_checkpoint.decoder[:ACTIONS] - base_checkpoint.decoder[:ACTIONS]
    ).double()
    projected_delta = source_delta @ projector
    early_output_rms = torch.sqrt(
        torch.clamp(
            torch.einsum("ah,hk,ak->a", projected_delta, covariance, projected_delta),
            min=0.0,
        )
    )

    serialized = np.fromfile(args.base, dtype=np.float32)
    decoder_offset = _align8(base_checkpoint.hidden_dim * base_checkpoint.input_dim)
    action_count = ACTIONS * base_checkpoint.hidden_dim
    serialized[decoder_offset:decoder_offset + action_count] = (
        base_checkpoint.decoder[:ACTIONS]
        + projected_delta.float().numpy()
    ).reshape(-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized.tofile(args.output)
    CheckpointPolicy.load(str(args.output))

    retained = float(torch.linalg.norm(projected_delta) / torch.linalg.norm(source_delta))
    print(
        f"wrote={args.output} early_samples={samples} "
        f"retained_dimensions={int(selected.sum())}/{model.hidden_dim} "
        f"threshold={args.max_early_eigenvalue:.9g} retained_delta_l2={retained:.9f} "
        f"early_output_rms=({','.join(f'{value:.9g}' for value in early_output_rms.tolist())})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
