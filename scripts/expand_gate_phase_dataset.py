#!/usr/bin/env python3
"""Build 32-input demonstrations with observable phase-gated features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


TARGET_OBSERVATIONS = 32
ACTIONS = 4


def expand_records(source: np.ndarray, source_observations: int) -> np.ndarray:
    source_width = source_observations + ACTIONS + 1
    if source.ndim != 2 or source.shape[1] != source_width:
        raise ValueError(f"expected source shape (N, {source_width})")
    target_width = TARGET_OBSERVATIONS + ACTIONS + 1
    target = np.zeros((len(source), target_width), dtype=np.float32)
    target[:, :26] = source[:, :26]

    gate_two = source[:, 24]
    gate_three = source[:, 25]
    target[:, 26] = gate_two * source[:, 12]
    target[:, 27] = gate_two * source[:, 1]
    target[:, 28] = gate_two * source[:, 13]
    target[:, 29] = gate_two * source[:, 2]
    target[:, 30] = gate_three * source[:, 12]
    target[:, 31] = gate_three * source[:, 1]
    target[:, TARGET_OBSERVATIONS : TARGET_OBSERVATIONS + ACTIONS] = source[
        :, source_observations : source_observations + ACTIONS
    ]
    target[:, -1] = source[:, -1]
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-observations", type=int, choices=(26, 32), default=26)
    args = parser.parse_args()

    source_observations = args.source_observations
    source_width = source_observations + ACTIONS + 1
    target_width = TARGET_OBSERVATIONS + ACTIONS + 1
    values = np.fromfile(args.input, dtype=np.float32)
    if values.size % source_width:
        raise ValueError(f"{args.input} contains a partial source record")
    source = values.reshape(-1, source_width)
    target = expand_records(source, source_observations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    target.tofile(args.output)
    print(f"wrote {args.output} records={len(target)} width={target_width}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
