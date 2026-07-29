#!/usr/bin/env python3
"""Scale a bounded block of encoder hidden rows in a raw policy checkpoint.

This is an explicit low-dimensional recurrent-feature calibration primitive.
It preserves the checkpoint ABI, decoder, MinGRU weights, and runtime control.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy


def encoder_row_bounds(
    *, input_dim: int, hidden_dim: int, row_start: int, row_count: int
) -> tuple[int, int]:
    if row_start < 0 or row_count < 1 or row_start + row_count > hidden_dim:
        raise ValueError("encoder row block is outside the hidden dimension")
    value_start = row_start * input_dim
    return value_start, value_start + row_count * input_dim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--row-start", type=int, required=True)
    parser.add_argument("--row-count", type=int, default=4)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    args = parser.parse_args()

    if not np.isfinite(args.scale) or args.scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    CheckpointPolicy.load(
        str(args.input),
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
    )
    value_start, value_end = encoder_row_bounds(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        row_start=args.row_start,
        row_count=args.row_count,
    )
    weights = np.fromfile(args.input, dtype=np.float32)
    before_norm = float(np.linalg.norm(weights[value_start:value_end]))
    weights[value_start:value_end] *= np.float32(args.scale)
    after_norm = float(np.linalg.norm(weights[value_start:value_end]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights.tofile(args.output)
    CheckpointPolicy.load(
        str(args.output),
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
    )
    print(
        f"wrote {args.output} rows={args.row_start}:"
        f"{args.row_start + args.row_count} scale={args.scale} "
        f"block_norm={before_norm:.6f}->{after_norm:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
