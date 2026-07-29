#!/usr/bin/env python3
"""Blend deployed parameters between two PufferNet checkpoints.

The default output keeps every byte from the base checkpoint except the four
policy action rows. ``--scope all`` interpolates the complete checkpoint and is
useful for line-searching a constrained recurrent update.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy, _align8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interpolate PufferNet action-decoder rows without changing recurrent weights."
    )
    parser.add_argument("base", type=Path, help="Officially validated source checkpoint")
    parser.add_argument("candidate", type=Path, help="Late-course decoder candidate")
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--action-alphas",
        type=float,
        nargs=4,
        metavar=("PITCH", "ROLL", "THRUST", "YAW"),
        help="Optional per-action residual scales; overrides --alpha for each action row.",
    )
    parser.add_argument(
        "--scope", choices=("action-decoder", "all"), default="action-decoder"
    )
    parser.add_argument("--input-dim", type=int, default=23)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument(
        "--layout-precision-bytes", type=int, choices=(2, 4), default=2,
        help="Native checkpoint arena precision (2=BF16 layout, 4=FP32 layout).",
    )
    args = parser.parse_args()

    if not np.isfinite(args.alpha) or not -8.0 <= args.alpha <= 8.0:
        parser.error("--alpha must be finite and in [-8, 8]")
    if args.action_alphas is not None and any(
        not np.isfinite(value) or not -8.0 <= value <= 8.0
        for value in args.action_alphas
    ):
        parser.error("--action-alphas values must be finite and in [-8, 8]")

    load_kwargs = {
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "num_actions": args.num_actions,
        "layout_precision_bytes": args.layout_precision_bytes,
    }
    base_policy = CheckpointPolicy.load(str(args.base), **load_kwargs)
    candidate_policy = CheckpointPolicy.load(str(args.candidate), **load_kwargs)
    if base_policy.num_layers != candidate_policy.num_layers:
        raise ValueError("base and candidate use different recurrent depths")

    base = np.fromfile(args.base, dtype=np.float32)
    candidate = np.fromfile(args.candidate, dtype=np.float32)
    if base.shape != candidate.shape:
        raise ValueError(
            f"checkpoint sizes differ: base={base.size} candidate={candidate.size} floats"
        )

    decoder_offset = _align8(args.hidden_dim * args.input_dim)
    action_count = args.num_actions * args.hidden_dim
    action_slice = slice(decoder_offset, decoder_offset + action_count)
    if args.scope == "all":
        output = base + np.float32(args.alpha) * (candidate - base)
    else:
        output = base.copy()
        if args.action_alphas is None:
            output[action_slice] = (
                base[action_slice]
                + np.float32(args.alpha) * (candidate[action_slice] - base[action_slice])
            )
        else:
            base_rows = base[action_slice].reshape(args.num_actions, args.hidden_dim)
            candidate_rows = candidate[action_slice].reshape(
                args.num_actions, args.hidden_dim
            )
            output[action_slice] = (
                base_rows
                + np.asarray(args.action_alphas, dtype=np.float32)[:, None]
                * (candidate_rows - base_rows)
            ).reshape(-1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(args.output)
    blended_policy = CheckpointPolicy.load(str(args.output), **load_kwargs)

    if args.scope == "action-decoder":
        outside = np.ones(base.size, dtype=bool)
        outside[action_slice] = False
        if not np.array_equal(output[outside], base[outside]):
            raise RuntimeError("output changed checkpoint parameters outside action decoder")

    delta = candidate_policy.decoder[: args.num_actions] - base_policy.decoder[: args.num_actions]
    applied = blended_policy.decoder[: args.num_actions] - base_policy.decoder[: args.num_actions]
    print(
        f"wrote={args.output} scope={args.scope} alpha={args.alpha:.6f} "
        f"action_alphas={args.action_alphas} "
        f"candidate_delta_l2={np.linalg.norm(delta):.9f} "
        f"applied_delta_l2={np.linalg.norm(applied):.9f} "
        f"base_sha256={_sha256(args.base)} output_sha256={_sha256(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
