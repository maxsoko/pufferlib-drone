#!/usr/bin/env python3
"""Scale one PufferNet encoder input column without changing any other tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from policy_callable_checkpoint import CheckpointPolicy
except ModuleNotFoundError:  # Imported as scripts.scale_policy_encoder_feature.
    from scripts.policy_callable_checkpoint import CheckpointPolicy


def scale_encoder_feature(
    input_path: Path,
    output_path: Path,
    *,
    feature_index: int,
    scale: float,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_actions: int = 4,
    layout_precision_bytes: int = 4,
) -> tuple[float, float]:
    if not 0 <= feature_index < input_dim:
        raise ValueError("feature index is outside the encoder input")
    if not np.isfinite(scale):
        raise ValueError("scale must be finite")

    CheckpointPolicy.load(
        str(input_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
        layout_precision_bytes=layout_precision_bytes,
    )
    serialized = np.fromfile(input_path, dtype=np.float32)
    encoder_count = hidden_dim * input_dim
    if serialized.size < encoder_count:
        raise ValueError("checkpoint ended before the encoder")

    output = serialized.copy()
    encoder = output[:encoder_count].reshape(hidden_dim, input_dim)
    before_norm = float(np.linalg.norm(encoder[:, feature_index]))
    encoder[:, feature_index] *= np.float32(scale)
    after_norm = float(np.linalg.norm(encoder[:, feature_index]))

    changed = output != serialized
    allowed = np.zeros_like(changed)
    allowed[:encoder_count].reshape(hidden_dim, input_dim)[:, feature_index] = True
    if np.any(changed & ~allowed):
        raise RuntimeError("scaling changed parameters outside the selected encoder column")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(output_path)
    CheckpointPolicy.load(
        str(output_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
        layout_precision_bytes=layout_precision_bytes,
    )
    return before_norm, after_norm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--feature-index", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument(
        "--layout-precision-bytes", type=int, choices=(2, 4), default=4
    )
    args = parser.parse_args()

    try:
        before_norm, after_norm = scale_encoder_feature(
            args.input,
            args.output,
            feature_index=args.feature_index,
            scale=args.scale,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_actions=args.num_actions,
            layout_precision_bytes=args.layout_precision_bytes,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"wrote={args.output} feature={args.feature_index} scale={args.scale} "
        f"column_norm={before_norm:.9f}->{after_norm:.9f} "
        f"layout_precision_bytes={args.layout_precision_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
