#!/usr/bin/env python3
"""Move one learned PufferNet input contribution into another feature.

This is useful when an observation-contract correction changes the numerical
meaning of a feature that acted as a nearly constant bias during training.  It
preserves the recurrent network, decoder, exploration parameters, and every
unrelated encoder column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from policy_callable_checkpoint import CheckpointPolicy
except ModuleNotFoundError:  # Imported as scripts.remap_policy_encoder_feature.
    from scripts.policy_callable_checkpoint import CheckpointPolicy


def remap(
    input_path: Path,
    output_path: Path,
    *,
    source_index: int,
    target_index: int,
    transfer_scale: float,
    source_retain_scale: float,
    input_dim: int,
    hidden_dim: int,
    num_actions: int,
) -> tuple[float, float, float]:
    if not 0 <= source_index < input_dim:
        raise ValueError("source index is outside the encoder input")
    if not 0 <= target_index < input_dim:
        raise ValueError("target index is outside the encoder input")
    if source_index == target_index:
        raise ValueError("source and target indices must differ")
    if not np.isfinite(transfer_scale) or not np.isfinite(source_retain_scale):
        raise ValueError("feature scales must be finite")

    serialized = np.fromfile(input_path, dtype=np.float32)
    encoder_count = hidden_dim * input_dim
    if serialized.size < encoder_count:
        raise ValueError("checkpoint ended before the encoder")
    output = serialized.copy()
    encoder = output[:encoder_count].reshape(hidden_dim, input_dim)
    source = encoder[:, source_index].copy()
    source_norm = float(np.linalg.norm(source))
    encoder[:, target_index] += np.float32(transfer_scale) * source
    encoder[:, source_index] = np.float32(source_retain_scale) * source

    changed = output != serialized
    allowed = np.zeros_like(changed)
    allowed[:encoder_count].reshape(hidden_dim, input_dim)[:, source_index] = True
    allowed[:encoder_count].reshape(hidden_dim, input_dim)[:, target_index] = True
    if np.any(changed & ~allowed):
        raise RuntimeError("remap changed parameters outside the declared encoder columns")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(output_path)
    loaded = CheckpointPolicy.load(
        str(output_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
    )
    return (
        source_norm,
        float(np.linalg.norm(loaded.encoder[:, source_index])),
        float(np.linalg.norm(loaded.encoder[:, target_index])),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--transfer-scale", type=float, default=1.0)
    parser.add_argument("--source-retain-scale", type=float, default=0.0)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    args = parser.parse_args()

    try:
        source_before, source_after, target_after = remap(
            args.input,
            args.output,
            source_index=args.source_index,
            target_index=args.target_index,
            transfer_scale=args.transfer_scale,
            source_retain_scale=args.source_retain_scale,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_actions=args.num_actions,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"wrote={args.output} source={args.source_index} target={args.target_index} "
        f"transfer_scale={args.transfer_scale} retain_scale={args.source_retain_scale} "
        f"source_norm={source_before:.9f}->{source_after:.9f} "
        f"target_norm={target_after:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
