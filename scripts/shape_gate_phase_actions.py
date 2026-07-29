#!/usr/bin/env python3
"""Apply a bounded action-label correction to one observable gate phase."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from train_full_policy_bc import ACTIONS, OBSERVATIONS, RECORD_WIDTH


def gate_phase_mask(records: np.ndarray, gate_index: int) -> np.ndarray:
    if records.ndim != 2 or records.shape[1] != RECORD_WIDTH:
        raise ValueError(f"records must have shape [N, {RECORD_WIDTH}]")
    if not 0 <= gate_index <= 3:
        raise ValueError("gate_index must be in [0, 3]")
    phases = records[:, 23:26]
    if gate_index == 0:
        return np.all(phases < 0.5, axis=1)
    return phases[:, gate_index - 1] > 0.5


def shape_gate_action(
    records: np.ndarray,
    *,
    gate_index: int,
    action_index: int,
    scale: float,
    offset: float,
    min_apparent_size: float | None = None,
    full_apparent_size: float | None = None,
) -> int:
    if not 0 <= action_index < ACTIONS:
        raise ValueError(f"action_index must be in [0, {ACTIONS - 1}]")
    if not np.isfinite(scale) or not np.isfinite(offset):
        raise ValueError("scale and offset must be finite")
    mask = gate_phase_mask(records, gate_index)
    ramp = np.ones(len(records), dtype=np.float32)
    if min_apparent_size is not None or full_apparent_size is not None:
        if min_apparent_size is None or full_apparent_size is None:
            raise ValueError("both apparent-size bounds are required")
        if full_apparent_size <= min_apparent_size:
            raise ValueError("full apparent size must exceed minimum apparent size")
        ramp = np.clip(
            (records[:, 16] - np.float32(min_apparent_size))
            / np.float32(full_apparent_size - min_apparent_size),
            0.0,
            1.0,
        )
        mask &= ramp > 0.0
    column = OBSERVATIONS + action_index
    records[mask, column] = np.clip(
        records[mask, column]
        * (1.0 + np.float32(scale - 1.0) * ramp[mask])
        + np.float32(offset) * ramp[mask],
        -1.0,
        1.0,
    )
    return int(mask.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gate-index", type=int, required=True)
    parser.add_argument("--action-index", type=int, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--min-apparent-size", type=float, default=None)
    parser.add_argument("--full-apparent-size", type=float, default=None)
    args = parser.parse_args()

    values = np.fromfile(args.input, dtype=np.float32)
    if values.size % RECORD_WIDTH:
        raise ValueError(f"{args.input} contains a partial teacher record")
    records = values.reshape(-1, RECORD_WIDTH).copy()
    shaped = shape_gate_action(
        records,
        gate_index=args.gate_index,
        action_index=args.action_index,
        scale=args.scale,
        offset=args.offset,
        min_apparent_size=args.min_apparent_size,
        full_apparent_size=args.full_apparent_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records.tofile(args.output)
    print(
        f"wrote={args.output} records={len(records)} shaped={shaped} "
        f"gate={args.gate_index} action={args.action_index} "
        f"scale={args.scale} offset={args.offset}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
