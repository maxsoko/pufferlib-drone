#!/usr/bin/env python3
"""Create a stable full PufferNet policy from the observable gate-motion expert.

This is a policy checkpoint, not a runtime controller: the encoder exposes the
observable features, MinGRU layers start as near-identity pass-throughs, and the
decoder contains the proportional expert. PPO or sequence BC can fine-tune all
weights from this well-conditioned starting point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from policy_callable_checkpoint import CheckpointPolicy, _align


def write_tensor(
    serialized: np.ndarray,
    offset: int,
    tensor: np.ndarray,
    *,
    precision_bytes: int,
) -> int:
    flat = np.asarray(tensor, dtype=np.float32).reshape(-1)
    available = max(0, min(len(flat), len(serialized) - offset))
    if available:
        serialized[offset:offset + available] = flat[:available]
    return _align(offset + len(flat), precision_bytes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Source checkpoint supplying the native layout")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--input-dim",
        type=int,
        choices=(23, 32),
        default=23,
        help="Observation width encoded in the source checkpoint (default: 23).",
    )
    parser.add_argument(
        "--checkpoint-layout-precision-bytes",
        type=int,
        choices=(2, 4),
        default=2,
        help="Native arena alignment used by the source and output checkpoints.",
    )
    parser.add_argument("--forward-speed", type=float, default=2.0)
    parser.add_argument("--forward-kp", type=float, default=0.35)
    parser.add_argument("--right-position-kp", type=float, default=0.08)
    parser.add_argument("--right-rate-kp", type=float, default=0.10)
    parser.add_argument("--down-position-kp", type=float, default=0.08)
    parser.add_argument("--down-rate-kp", type=float, default=0.25)
    parser.add_argument(
        "--down-alignment-pitch-action",
        type=float,
        default=0.0,
        help="Positive pitch-action coefficient on normalized gate-down error to slow while unaligned",
    )
    parser.add_argument("--yaw-error-kp", type=float, default=0.0)
    parser.add_argument(
        "--yaw-attitude-feedback",
        action="store_true",
        help="Add the current small-angle quaternion yaw to the absolute-yaw output",
    )
    parser.add_argument("--highway-logit", type=float, default=-10.0)
    parser.add_argument(
        "--log-std",
        type=float,
        default=-20.0,
        help=(
            "Gaussian action log standard deviation. Keep -20 for deterministic "
            "deployment checkpoints; use about -3 for a PPO training initializer."
        ),
    )
    parser.add_argument("--pitch-action-scale", type=float, default=0.24)
    parser.add_argument("--roll-action-scale", type=float, default=0.40)
    parser.add_argument(
        "--linear-gate-features",
        action="store_true",
        help="Initialize for clipped linear gate features instead of tanh features.",
    )
    args = parser.parse_args()
    if args.forward_speed <= 0.0:
        parser.error("--forward-speed must be positive")

    source = CheckpointPolicy.load(
        str(args.input),
        input_dim=args.input_dim,
        layout_precision_bytes=args.checkpoint_layout_precision_bytes,
    )
    if source.input_dim not in (23, 32) or source.hidden_dim != 128 or source.num_actions != 4:
        raise ValueError(
            "observable teacher initializer expects a 23- or 32-input 128x4 race policy"
        )

    encoder = np.zeros_like(source.encoder)
    # Channel 0 is a bias carrier from quaternion w (normally approximately 1).
    encoder[0, 6] = 1.0
    # Channels 1..23 expose every observation without entangling features.
    for observation_index in range(source.input_dim):
        encoder[observation_index + 1, observation_index] = 1.0

    mingru: list[np.ndarray] = []
    for _ in range(source.num_layers):
        projection = np.zeros((3 * source.hidden_dim, source.hidden_dim), dtype=np.float32)
        # A large negative highway logit selects the direct input path. The
        # recurrent candidate remains available for subsequent learning.
        projection[2 * source.hidden_dim:3 * source.hidden_dim, 0] = args.highway_logit
        mingru.append(projection)

    decoder = np.zeros_like(source.decoder)
    # Feature channel N+1 corresponds to observation N.
    forward_rate = 1
    right_rate = 2
    down_rate = 3
    right_position = 13
    down_position = 14
    yaw_error = 15
    quat_z = 10

    # camera gate-rate obs uses tanh(rate/scale); linearization is intentionally
    # conservative and remains inside the trained flight envelope.
    decoder[0, 0] = -args.pitch_action_scale * args.forward_speed * args.forward_kp
    if args.linear_gate_features:
        decoder[0, forward_rate] = -5.0 * args.pitch_action_scale * args.forward_kp
    else:
        encoded_target_rate = np.tanh(-args.forward_speed * 0.2)
        decoder[0, forward_rate] = (
            args.pitch_action_scale * args.forward_speed * args.forward_kp
            / encoded_target_rate
        )
    decoder[0, down_position] += args.down_alignment_pitch_action
    decoder[1, right_rate] = -args.roll_action_scale * 3.0 * args.right_rate_kp
    decoder[1, right_position] = -args.roll_action_scale * 5.0 * args.right_position_kp
    decoder[2, down_rate] = -3.0 * args.down_rate_kp
    decoder[2, down_position] = -5.0 * args.down_position_kp
    # Mode 2 outputs absolute yaw. For the small-angle Stage-C envelope,
    # yaw ~= 2*qz and desired_yaw = yaw + kp*camera_yaw_error.
    decoder[3, quat_z] = 2.0 / np.pi if args.yaw_attitude_feedback else 0.0
    decoder[3, yaw_error] = 0.25 * args.yaw_error_kp

    serialized = np.fromfile(args.input, dtype=np.float32)
    serialized.fill(0.0)
    offset = 0
    offset = write_tensor(
        serialized,
        offset,
        encoder,
        precision_bytes=args.checkpoint_layout_precision_bytes,
    )
    offset = write_tensor(
        serialized,
        offset,
        decoder,
        precision_bytes=args.checkpoint_layout_precision_bytes,
    )
    offset = write_tensor(
        serialized,
        offset,
        np.full(4, args.log_std, dtype=np.float32),
        precision_bytes=args.checkpoint_layout_precision_bytes,
    )
    for projection in mingru:
        offset = write_tensor(
            serialized,
            offset,
            projection,
            precision_bytes=args.checkpoint_layout_precision_bytes,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized.tofile(args.output)
    CheckpointPolicy.load(
        str(args.output),
        input_dim=args.input_dim,
        layout_precision_bytes=args.checkpoint_layout_precision_bytes,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
