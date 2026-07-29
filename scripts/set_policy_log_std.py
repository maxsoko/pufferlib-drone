#!/usr/bin/env python3
"""Set continuous-action exploration log standard deviation in a raw checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from policy_callable_checkpoint import CheckpointPolicy, _align
except ModuleNotFoundError:  # Imported as scripts.set_policy_log_std.
    from scripts.policy_callable_checkpoint import CheckpointPolicy, _align


def set_log_std(
    input_path: Path,
    output_path: Path,
    *,
    log_std: float,
    input_dim: int,
    hidden_dim: int = 128,
    num_actions: int = 4,
    layout_precision_bytes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace only the continuous-action log standard deviations."""
    if not np.isfinite(log_std):
        raise ValueError("log_std must be finite")
    if min(input_dim, hidden_dim, num_actions) < 1:
        raise ValueError("checkpoint dimensions must be positive")

    CheckpointPolicy.load(
        str(input_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
        layout_precision_bytes=layout_precision_bytes,
    )
    weights = np.fromfile(input_path, dtype=np.float32)
    encoder_end = _align(hidden_dim * input_dim, layout_precision_bytes)
    decoder_end = _align(
        encoder_end + (num_actions + 1) * hidden_dim,
        layout_precision_bytes,
    )
    log_std_end = decoder_end + num_actions
    if log_std_end > weights.size:
        raise ValueError("checkpoint ended before continuous-action log_std")
    before = weights[decoder_end:log_std_end].copy()
    weights[decoder_end:log_std_end] = np.float32(log_std)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights.tofile(output_path)
    loaded = CheckpointPolicy.load(
        str(output_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
        layout_precision_bytes=layout_precision_bytes,
    )
    return before, loaded.log_std.copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--log-std", type=float, required=True)
    parser.add_argument(
        "--input-dim",
        type=int,
        required=True,
        help=(
            "Checkpoint observation width. This is required because using the "
            "wrong legacy/full-policy width edits encoder weights instead of log_std."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument(
        "--layout-precision-bytes", type=int, choices=(2, 4), default=2,
        help="Native arena alignment used by the checkpoint (default: 2).",
    )
    args = parser.parse_args()
    try:
        before, after = set_log_std(
            args.input,
            args.output,
            log_std=args.log_std,
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_actions=args.num_actions,
            layout_precision_bytes=args.layout_precision_bytes,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"wrote={args.output} log_std_before={before.tolist()} "
        f"log_std_after={after.tolist()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
