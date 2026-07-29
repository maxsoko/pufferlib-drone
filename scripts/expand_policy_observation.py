#!/usr/bin/env python3
"""Append zero-initialized observable inputs to a recurrent PufferNet checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy, _align8


def _serialize(
    policy: CheckpointPolicy,
    *,
    target_input_dim: int,
) -> np.ndarray:
    if target_input_dim < policy.input_dim:
        raise ValueError("target input dimension must not shrink the policy")

    encoder = np.zeros((policy.hidden_dim, target_input_dim), dtype=np.float32)
    encoder[:, : policy.input_dim] = policy.encoder
    tensors = [
        encoder,
        policy.decoder,
        policy.log_std,
        *policy.mingru_proj,
    ]
    logical_count = sum(tensor.size for tensor in tensors)
    serialized = np.zeros(logical_count, dtype=np.float32)
    offset = 0
    for tensor in tensors:
        values = np.asarray(tensor, dtype=np.float32).reshape(-1)
        available = min(values.size, serialized.size - offset)
        if available > 0:
            serialized[offset : offset + available] = values[:available]
        offset = _align8(offset + values.size)
    return serialized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-input-dim", type=int, default=23)
    parser.add_argument("--target-input-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=4)
    args = parser.parse_args()

    if args.source_input_dim < 1 or args.target_input_dim <= args.source_input_dim:
        parser.error("target input dimension must be larger than source input dimension")

    source = CheckpointPolicy.load(
        str(args.input),
        input_dim=args.source_input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_actions=args.num_actions,
    )
    serialized = _serialize(source, target_input_dim=args.target_input_dim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized.tofile(args.output)

    expanded = CheckpointPolicy.load(
        str(args.output),
        input_dim=args.target_input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_actions=args.num_actions,
    )
    rng = np.random.default_rng(3385)
    max_action_error = 0.0
    for _ in range(32):
        observation = rng.uniform(-1.0, 1.0, args.source_input_dim).astype(np.float32)
        source_action = np.asarray(source.infer(observation), dtype=np.float32)
        expanded_observation = np.zeros(args.target_input_dim, dtype=np.float32)
        expanded_observation[: args.source_input_dim] = observation
        expanded_action = np.asarray(expanded.infer(expanded_observation), dtype=np.float32)
        max_action_error = max(
            max_action_error,
            float(np.max(np.abs(source_action - expanded_action))),
        )
    if max_action_error > 1e-6:
        raise RuntimeError(f"expanded checkpoint changed base actions by {max_action_error}")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"wrote {args.output} input_dim={args.source_input_dim}->{args.target_input_dim} "
        f"max_action_error={max_action_error:.9g} sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
