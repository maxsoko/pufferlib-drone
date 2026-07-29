#!/usr/bin/env python3
"""Create live-detector-like variants of full-policy teacher trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from train_full_policy_bc import OBSERVATIONS, RECORD_WIDTH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--from-race-phase", type=float, default=2.0 / 3.0)
    parser.add_argument("--confidence-scale", type=float, required=True)
    parser.add_argument("--size-scale", type=float, required=True)
    parser.add_argument("--rate-noise-std", type=float, default=0.0)
    parser.add_argument("--elapsed-offset", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3385)
    args = parser.parse_args()
    if not 0.0 <= args.from_race_phase <= 1.0:
        parser.error("from-race-phase must be in [0, 1]")
    if args.confidence_scale < 0.0 or args.size_scale < 0.0:
        parser.error("confidence and size scales must be nonnegative")
    if args.rate_noise_std < 0.0:
        parser.error("rate-noise-std must be nonnegative")

    values = np.fromfile(args.input, dtype=np.float32)
    if values.size % RECORD_WIDTH:
        raise ValueError("input contains a partial teacher record")
    records = values.reshape(-1, RECORD_WIDTH).copy()
    active = records[:, 22] >= np.float32(args.from_race_phase)
    observations = records[:, :OBSERVATIONS]
    rng = np.random.default_rng(args.seed)
    if args.rate_noise_std:
        noise = rng.normal(
            0.0, args.rate_noise_std, size=(int(active.sum()), 3)
        ).astype(np.float32)
        observations[active, 0:3] = np.clip(
            observations[active, 0:3] + noise, -1.0, 1.0
        )
    observations[active, 16] = np.clip(
        observations[active, 16] * np.float32(args.size_scale), 0.0, 1.0
    )
    observations[active, 17] = np.clip(
        observations[active, 17] * np.float32(args.confidence_scale), 0.0, 1.0
    )
    observations[active, 18] = np.clip(
        observations[active, 18] + np.float32(args.elapsed_offset), 0.0, 1.0
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records.tofile(args.output)
    print(
        f"wrote={args.output} records={len(records)} active={int(active.sum())} "
        f"confidence_scale={args.confidence_scale} size_scale={args.size_scale} "
        f"rate_noise_std={args.rate_noise_std} elapsed_offset={args.elapsed_offset}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
