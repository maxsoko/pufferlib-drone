#!/usr/bin/env python3
"""Repack a raw PufferNet checkpoint between native precision alignments.

Checkpoints always store FP32 master values, but tensor starts mirror the
16-byte alignment of the native parameter arena. BF16 therefore aligns tensor
starts to 8 serialized floats while an FP32 build aligns them to 4. Loading a
checkpoint written for one layout directly into the other shifts tensors after
the first non-aligned parameter (the continuous-action ``log_std`` tensor).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _align(index: int, precision_bytes: int) -> int:
    alignment_elements = 16 // precision_bytes
    return (index + alignment_elements - 1) & ~(alignment_elements - 1)


def tensor_counts(
    *, input_dim: int, hidden_dim: int, num_layers: int, num_actions: int
) -> list[int]:
    return [
        hidden_dim * input_dim,
        (num_actions + 1) * hidden_dim,
        num_actions,
        *([3 * hidden_dim * hidden_dim] * num_layers),
    ]


def unpack_layout(
    serialized: np.ndarray, counts: list[int], *, precision_bytes: int
) -> list[np.ndarray]:
    logical_count = sum(counts)
    if serialized.size != logical_count:
        raise ValueError(
            f"checkpoint size mismatch: expected {logical_count} floats, "
            f"got {serialized.size}"
        )
    aligned_count = 0
    for count in counts:
        aligned_count = _align(aligned_count + count, precision_bytes)
    storage = np.pad(
        np.asarray(serialized, dtype=np.float32),
        (0, max(0, aligned_count - serialized.size)),
        mode="constant",
    )
    tensors: list[np.ndarray] = []
    index = 0
    for count in counts:
        tensor = storage[index:index + count]
        if tensor.size != count:
            raise ValueError("checkpoint ended inside an aligned tensor")
        tensors.append(tensor.copy())
        index = _align(index + count, precision_bytes)
    return tensors


def pack_layout(
    tensors: list[np.ndarray], counts: list[int], *, precision_bytes: int
) -> np.ndarray:
    logical_count = sum(counts)
    aligned_count = 0
    for count in counts:
        aligned_count = _align(aligned_count + count, precision_bytes)
    storage = np.zeros(max(logical_count, aligned_count), dtype=np.float32)
    index = 0
    for tensor, count in zip(tensors, counts, strict=True):
        flat = np.asarray(tensor, dtype=np.float32).reshape(-1)
        if flat.size != count:
            raise ValueError(f"tensor size mismatch: expected {count}, got {flat.size}")
        storage[index:index + count] = flat
        index = _align(index + count, precision_bytes)
    # Native save/load uses the unpadded logical element count. Any aligned tail
    # is consequently absent and is reconstructed as zeros while unpacking.
    return storage[:logical_count]


def convert_layout(
    serialized: np.ndarray,
    *,
    source_precision_bytes: int,
    target_precision_bytes: int,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_actions: int,
) -> np.ndarray:
    for name, value in (
        ("source_precision_bytes", source_precision_bytes),
        ("target_precision_bytes", target_precision_bytes),
    ):
        if value not in (2, 4):
            raise ValueError(f"{name} must be 2 (BF16) or 4 (FP32)")
    counts = tensor_counts(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )
    tensors = unpack_layout(
        serialized, counts, precision_bytes=source_precision_bytes
    )
    return pack_layout(tensors, counts, precision_bytes=target_precision_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-precision-bytes", type=int, required=True)
    parser.add_argument("--target-precision-bytes", type=int, required=True)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=4)
    args = parser.parse_args()

    serialized = np.fromfile(args.input, dtype=np.float32)
    converted = convert_layout(
        serialized,
        source_precision_bytes=args.source_precision_bytes,
        target_precision_bytes=args.target_precision_bytes,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_actions=args.num_actions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.tofile(args.output)
    print(
        f"wrote {args.output} floats={converted.size} "
        f"layout={args.source_precision_bytes}->{args.target_precision_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
