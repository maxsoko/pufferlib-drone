#!/usr/bin/env python3
"""Crop every 32-observation recurrent teacher episode to an early prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


OBSERVATIONS = 32
ACTIONS = 4
RECORD_WIDTH = OBSERVATIONS + ACTIONS + 1


def crop_episode_prefixes(records: np.ndarray, max_steps: int) -> np.ndarray:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    records = np.asarray(records, dtype=np.float32)
    if records.ndim != 2 or records.shape[1] != RECORD_WIDTH:
        raise ValueError(f"expected records with width {RECORD_WIDTH}")
    starts = np.flatnonzero(records[:, -1] > 0.5)
    if not len(starts) or starts[0] != 0:
        raise ValueError("dataset must begin with an episode reset record")
    ends = np.r_[starts[1:], len(records)]
    prefixes = [records[start : min(end, start + max_steps)] for start, end in zip(starts, ends)]
    return np.concatenate(prefixes)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--json-path", type=Path)
    args = parser.parse_args()

    values = np.fromfile(args.input, dtype=np.float32)
    if values.size % RECORD_WIDTH:
        raise ValueError("input contains a partial teacher record")
    source = values.reshape(-1, RECORD_WIDTH)
    cropped = crop_episode_prefixes(source, args.max_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cropped.tofile(args.output)
    report = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "max_steps": args.max_steps,
        "episodes": int(np.count_nonzero(cropped[:, -1] > 0.5)),
        "source_records": int(len(source)),
        "output_records": int(len(cropped)),
        "record_width": RECORD_WIDTH,
    }
    json_path = args.json_path or args.output.with_suffix(args.output.suffix + ".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
