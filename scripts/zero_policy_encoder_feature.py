#!/usr/bin/env python3
"""Zero PufferNet encoder input columns while preserving checkpoint ABI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from policy_callable_checkpoint import CheckpointPolicy
except ModuleNotFoundError:  # Imported as scripts.zero_policy_encoder_feature.
    from scripts.policy_callable_checkpoint import CheckpointPolicy


def zero_encoder_features(
    input_path: Path,
    output_path: Path,
    *,
    feature_indices: list[int],
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_actions: int = 4,
    layout_precision_bytes: int = 4,
) -> tuple[dict[int, float], dict[int, float]]:
    selected = sorted(set(feature_indices))
    if not selected:
        raise ValueError("at least one feature index is required")
    if any(not 0 <= index < input_dim for index in selected):
        raise ValueError("feature index is outside the encoder input")

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
    before_norms = {
        index: float(np.linalg.norm(encoder[:, index])) for index in selected
    }
    encoder[:, selected] = 0.0

    changed = output != serialized
    allowed = np.zeros_like(changed)
    allowed[:encoder_count].reshape(hidden_dim, input_dim)[:, selected] = True
    if np.any(changed & ~allowed):
        raise RuntimeError("zeroing changed parameters outside selected encoder columns")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(output_path)
    loaded = CheckpointPolicy.load(
        str(output_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
        layout_precision_bytes=layout_precision_bytes,
    )
    after_norms = {
        index: float(np.linalg.norm(loaded.encoder[:, index])) for index in selected
    }
    if any(norm != 0.0 for norm in after_norms.values()):
        raise RuntimeError("an output encoder feature was not zeroed")
    return before_norms, after_norms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--feature-index", type=int, action="append", required=True,
        help="Encoder input column to zero; repeat for multiple columns.",
    )
    parser.add_argument("--input-dim", type=int, default=23)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--layout-precision-bytes", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    feature_indices = sorted(set(args.feature_index))
    try:
        before_norms, after_norms = zero_encoder_features(
            args.input,
            args.output,
            feature_indices=feature_indices,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_actions=args.num_actions,
            layout_precision_bytes=args.layout_precision_bytes,
        )
    except ValueError as exc:
        parser.error(str(exc))
    norm_summary = ",".join(
        f"{index}:{before_norms[index]:.9f}->{after_norms[index]:.9f}"
        for index in feature_indices
    )
    print(
        f"wrote={args.output} features={feature_indices} "
        f"column_norms={norm_summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
