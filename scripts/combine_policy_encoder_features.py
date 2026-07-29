#!/usr/bin/env python3
"""Combine independently trained encoder-column residuals into one checkpoint.

Every candidate must be byte-identical to the base outside its declared input
feature column. This permits signed, phase-gated line searches without changing
the decoder, recurrent weights, exploration parameters, or unrelated inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

try:
    from policy_callable_checkpoint import CheckpointPolicy
except ModuleNotFoundError:  # Imported as scripts.combine_policy_encoder_features.
    from scripts.policy_callable_checkpoint import CheckpointPolicy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_residual(values: list[str]) -> tuple[int, Path, float]:
    feature_text, checkpoint_text, scale_text = values
    try:
        feature = int(feature_text)
        scale = float(scale_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "feature residual must be FEATURE CHECKPOINT SCALE"
        ) from exc
    return feature, Path(checkpoint_text), scale


def combine(
    base_path: Path,
    output_path: Path,
    residuals: list[tuple[int, Path, float]],
    *,
    input_dim: int,
    hidden_dim: int,
    num_actions: int,
) -> list[tuple[int, float, float]]:
    base = np.fromfile(base_path, dtype=np.float32)
    output = base.copy()
    encoder_count = hidden_dim * input_dim
    base_encoder = base[:encoder_count].reshape(hidden_dim, input_dim)
    output_encoder = output[:encoder_count].reshape(hidden_dim, input_dim)
    summaries: list[tuple[int, float, float]] = []

    seen: set[int] = set()
    for feature, candidate_path, scale in residuals:
        if not 0 <= feature < input_dim:
            raise ValueError(f"feature {feature} is outside [0, {input_dim - 1}]")
        if feature in seen:
            raise ValueError(f"feature {feature} was specified more than once")
        if not math.isfinite(scale) or not -64.0 <= scale <= 64.0:
            raise ValueError("residual scale must be finite and in [-64, 64]")
        seen.add(feature)

        candidate = np.fromfile(candidate_path, dtype=np.float32)
        if candidate.shape != base.shape:
            raise ValueError(
                f"checkpoint sizes differ: base={base.size} candidate={candidate.size}"
            )
        candidate_encoder = candidate[:encoder_count].reshape(hidden_dim, input_dim)
        outside = np.ones(input_dim, dtype=bool)
        outside[feature] = False
        if not np.array_equal(candidate_encoder[:, outside], base_encoder[:, outside]):
            raise ValueError(
                f"candidate {candidate_path} changes encoder inputs outside feature {feature}"
            )
        if not np.array_equal(candidate[encoder_count:], base[encoder_count:]):
            raise ValueError(
                f"candidate {candidate_path} changes parameters outside the encoder"
            )

        delta = candidate_encoder[:, feature] - base_encoder[:, feature]
        output_encoder[:, feature] += np.float32(scale) * delta
        summaries.append(
            (feature, scale, float(np.linalg.norm(np.float32(scale) * delta)))
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(output_path)
    CheckpointPolicy.load(
        str(output_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
    )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--feature-residual",
        nargs=3,
        action="append",
        required=True,
        metavar=("FEATURE", "CHECKPOINT", "SCALE"),
    )
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    args = parser.parse_args()

    residuals = [parse_residual(values) for values in args.feature_residual]
    summaries = combine(
        args.base,
        args.output,
        residuals,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
    )
    print(
        f"wrote={args.output} residuals={summaries} "
        f"base_sha256={sha256(args.base)} output_sha256={sha256(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
