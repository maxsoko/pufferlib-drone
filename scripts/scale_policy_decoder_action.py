#!/usr/bin/env python3
"""Scale one continuous-action decoder row in a raw PufferLib checkpoint.

This is a deterministic behavior-cloning/calibration primitive for staged
curricula. It changes policy weights, not the runtime controller, and preserves
the checkpoint ABI and recurrent layers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy, _align8


def decoder_action_bounds(
    *, input_dim: int, hidden_dim: int, action_index: int
) -> tuple[int, int]:
    decoder_offset = _align8(hidden_dim * input_dim)
    row_start = decoder_offset + action_index * hidden_dim
    return row_start, row_start + hidden_dim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--action-index", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument(
        "--input-dim",
        type=int,
        default=32,
        help="Policy observation width (default: current full-policy ABI, 32).",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    args = parser.parse_args()

    if not 0 <= args.action_index < args.num_actions:
        raise ValueError("action-index is outside the decoder action rows")
    if not np.isfinite(args.scale):
        raise ValueError("scale must be finite")

    # Validate the source layout with the same parser used by the live runner.
    CheckpointPolicy.load(
        str(args.input),
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
    )
    weights = np.fromfile(args.input, dtype=np.float32)
    row_start, row_end = decoder_action_bounds(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        action_index=args.action_index,
    )
    if row_end > weights.size:
        raise ValueError("checkpoint ended before the requested decoder row")

    before_norm = float(np.linalg.norm(weights[row_start:row_end]))
    weights[row_start:row_end] *= np.float32(args.scale)
    after_norm = float(np.linalg.norm(weights[row_start:row_end]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights.tofile(args.output)
    CheckpointPolicy.load(
        str(args.output),
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
    )
    print(
        f"wrote {args.output} action={args.action_index} scale={args.scale} "
        f"row_norm={before_norm:.6f}->{after_norm:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
