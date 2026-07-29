#!/usr/bin/env python3
"""Behavior-clone roll/thrust decoder rows from official-observable inputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from drone_policy_contract import TS002_CAMERA_HALF_FOV_RAD, build_ts002_observation
from policy_callable_checkpoint import CheckpointPolicy, _align8


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trajectories", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--roll-per-m", type=float, default=0.03)
    parser.add_argument("--thrust-per-m", type=float, default=0.015)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3385)
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        help="Native teacher records: 23 observations, 4 actions, reset flag per row.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    model = CheckpointPolicy.load(str(args.input))
    features: list[np.ndarray] = []
    targets: list[tuple[float, float, float, float]] = []
    feature_dim = model.hidden_dim
    xtx = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    xty = np.zeros((feature_dim, 4), dtype=np.float64)
    yty = np.zeros(4, dtype=np.float64)
    sample_count = 0

    if args.dataset is not None:
        width = 23 + 4 + 1
        for dataset_path in args.dataset:
            records = np.fromfile(dataset_path, dtype=np.float32)
            if records.size % width:
                raise ValueError(f"{dataset_path} has a partial teacher record")
            records = records.reshape(-1, width)
            dataset_features = np.empty(
                (len(records), feature_dim), dtype=np.float32)
            model.reset_state()
            for row_index, record in enumerate(records):
                if record[-1] > 0.5:
                    model.reset_state()
                dataset_features[row_index] = model.hidden_features(record[:23])
            dataset_targets = records[:, 23:27]
            # Aggregate one dataset at a time so DAgger rounds do not require a
            # single, ever-growing feature matrix in memory.
            for block_start in range(0, len(records), 16384):
                block_end = min(block_start + 16384, len(records))
                feature_block = dataset_features[block_start:block_end].astype(
                    np.float64)
                target_block = dataset_targets[block_start:block_end].astype(
                    np.float64)
                xtx += feature_block.T @ feature_block
                xty += feature_block.T @ target_block
                yty += np.sum(target_block * target_block, axis=0)
            sample_count += len(records)

    for _ in range(0 if args.dataset is not None else args.trajectories):
        model.reset_state()
        forward0 = float(rng.uniform(3.0, 25.0))
        right0 = float(rng.uniform(-6.0, 6.0))
        down0 = float(rng.uniform(-2.0, 10.0))
        last_cmd = [float(rng.uniform(-0.05, 0.05)), 0.0, 0.0, 0.0]
        for step in range(args.sequence_length):
            phase = step / max(args.sequence_length - 1, 1)
            forward = max(0.5, forward0 * (1.0 - 0.8 * phase))
            right = right0 * (1.0 - 0.7 * phase)
            down = down0 * (1.0 - 0.7 * phase)
            yaw_error = math.atan2(right, max(forward, 1e-4))
            roll_target = clamp(right * args.roll_per_m, -0.35, 0.35)
            thrust_target = clamp(-down * args.thrust_per_m, -0.30, 0.15)
            observation = build_ts002_observation(
                guidance_visible=True,
                guidance_center_x_norm=clamp(
                    yaw_error / TS002_CAMERA_HALF_FOV_RAD, -1.0, 1.0),
                guidance_heading_error_rad=yaw_error,
                gate_visible=True,
                gate_forward_m=forward,
                gate_right_m=right,
                gate_down_m=down,
                gate_yaw_error_rad=yaw_error,
                gate_range_m=math.sqrt(forward * forward + right * right + down * down),
                elapsed_fraction=phase,
                last_cmd=last_cmd,
            )
            features.append(model.hidden_features(observation))
            targets.append((-0.50, roll_target, thrust_target, 0.0))
            last_cmd = [-0.50, roll_target, thrust_target, 0.0]

    if args.dataset is None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        xtx = x.T @ x
        xty = x.T @ y
        yty = np.sum(y * y, axis=0)
        sample_count = len(features)
    gram = xtx + args.ridge * np.eye(feature_dim, dtype=np.float64)
    solved_rows = np.linalg.solve(gram, xty).T
    decoder_rows = solved_rows.astype(np.float32)
    squared_error = np.empty(4, dtype=np.float64)
    for action_index, row in enumerate(solved_rows):
        squared_error[action_index] = (
            yty[action_index]
            - 2.0 * row @ xty[:, action_index]
            + row @ xtx @ row
        )
    rmse = np.sqrt(np.maximum(squared_error, 0.0) / sample_count)

    weights = np.fromfile(args.input, dtype=np.float32)
    decoder_offset = _align8(model.hidden_dim * model.input_dim)
    for action_index, row in enumerate(decoder_rows):
        start = decoder_offset + action_index * model.hidden_dim
        weights[start:start + model.hidden_dim] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights.tofile(args.output)
    CheckpointPolicy.load(str(args.output))
    print(
        f"wrote {args.output} samples={sample_count} "
        f"pitch_rmse={rmse[0]:.6f} roll_rmse={rmse[1]:.6f} "
        f"thrust_rmse={rmse[2]:.6f} yaw_rmse={rmse[3]:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
