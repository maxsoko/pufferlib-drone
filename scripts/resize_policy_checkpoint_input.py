#!/usr/bin/env python3
"""Expand a raw PufferNet checkpoint while preserving existing input behavior.

New encoder columns are initialized to zero. Therefore a 23-input recurrent
checkpoint preserves the same mathematical pre-update policy after expansion
to the 32-input six-gate progress ABI, regardless of the added input values.
Changing dot-product width/layout may introduce sub-microfloat rounding. The
tool also handles BF16/FP32 native arena alignment during the resize.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from convert_policy_checkpoint_layout import (
        pack_layout,
        tensor_counts,
        unpack_layout,
    )
except ModuleNotFoundError:  # Imported as scripts.resize_policy_checkpoint_input.
    from scripts.convert_policy_checkpoint_layout import (
        pack_layout,
        tensor_counts,
        unpack_layout,
    )


def expand_input(
    serialized: np.ndarray,
    *,
    source_input_dim: int,
    target_input_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_actions: int,
    source_precision_bytes: int,
    target_precision_bytes: int,
) -> np.ndarray:
    if source_input_dim <= 0 or target_input_dim < source_input_dim:
        raise ValueError("target_input_dim must be >= positive source_input_dim")
    source_counts = tensor_counts(
        input_dim=source_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )
    tensors = unpack_layout(
        serialized, source_counts, precision_bytes=source_precision_bytes
    )
    source_encoder = tensors[0].reshape(hidden_dim, source_input_dim)
    target_encoder = np.zeros((hidden_dim, target_input_dim), dtype=np.float32)
    target_encoder[:, :source_input_dim] = source_encoder
    target_tensors = [target_encoder.reshape(-1), *tensors[1:]]
    target_counts = tensor_counts(
        input_dim=target_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )
    return pack_layout(
        target_tensors, target_counts, precision_bytes=target_precision_bytes
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-input-dim", type=int, default=23)
    parser.add_argument("--target-input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--source-precision-bytes", type=int, default=2)
    parser.add_argument("--target-precision-bytes", type=int, default=4)
    args = parser.parse_args()

    output = expand_input(
        np.fromfile(args.input, dtype=np.float32),
        source_input_dim=args.source_input_dim,
        target_input_dim=args.target_input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_actions=args.num_actions,
        source_precision_bytes=args.source_precision_bytes,
        target_precision_bytes=args.target_precision_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(args.output)
    print(
        f"wrote {args.output} floats={output.size} "
        f"input={args.source_input_dim}->{args.target_input_dim} "
        f"layout={args.source_precision_bytes}->{args.target_precision_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
