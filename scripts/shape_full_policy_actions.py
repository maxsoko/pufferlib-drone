#!/usr/bin/env python3
"""Shape close-gate action labels in an existing full-policy BC dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from train_full_policy_bc import ACTIONS, OBSERVATIONS, RECORD_WIDTH


def shape_close_roll(
    records: np.ndarray,
    *,
    race_phase: float,
    min_apparent_size: float,
    full_apparent_size: float,
    roll_offset: float,
) -> int:
    if records.ndim != 2 or records.shape[1] != RECORD_WIDTH:
        raise ValueError(f"records must have shape [N, {RECORD_WIDTH}]")
    if full_apparent_size <= min_apparent_size:
        raise ValueError("full apparent size must exceed minimum apparent size")
    phase_mask = np.isclose(records[:, 22], race_phase, atol=1e-4)
    ramp = np.clip(
        (records[:, 16] - min_apparent_size)
        / (full_apparent_size - min_apparent_size),
        0.0,
        1.0,
    )
    mask = phase_mask & (ramp > 0.0)
    roll_index = OBSERVATIONS + 1
    records[mask, roll_index] = np.clip(
        records[mask, roll_index] + roll_offset * ramp[mask], -1.0, 1.0
    )
    return int(mask.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--race-phase", type=float, required=True)
    parser.add_argument("--min-apparent-size", type=float, default=0.25)
    parser.add_argument("--full-apparent-size", type=float, default=0.5)
    parser.add_argument("--roll-offset", type=float, required=True)
    args = parser.parse_args()

    values = np.fromfile(args.input, dtype=np.float32)
    if values.size % RECORD_WIDTH:
        raise ValueError(f"{args.input} contains a partial teacher record")
    records = values.reshape(-1, RECORD_WIDTH).copy()
    shaped = shape_close_roll(
        records,
        race_phase=args.race_phase,
        min_apparent_size=args.min_apparent_size,
        full_apparent_size=args.full_apparent_size,
        roll_offset=args.roll_offset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records.tofile(args.output)
    print(f"wrote={args.output} records={len(records)} shaped={shaped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
