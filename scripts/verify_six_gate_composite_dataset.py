#!/usr/bin/env python3
"""Replay an N142-style collector dataset through the six-gate callable."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


RECORD_WIDTH = 37


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(
    dataset: Path,
    *,
    infer_fn: Callable[[Sequence[float]], Sequence[float]],
    reset_fn: Callable[[], None],
    episodes: int,
    action_atol: float,
    required_gates: tuple[int, ...] = (3, 4, 5),
) -> dict:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    records = np.fromfile(dataset, dtype=np.float32)
    if records.size % RECORD_WIDTH:
        raise ValueError("dataset size is not divisible by the 37-float record width")
    records = records.reshape(-1, RECORD_WIDTH)
    starts = np.flatnonzero(records[:, 36] > 0.5)
    if starts.size < episodes:
        raise ValueError(
            f"dataset has {starts.size} episodes, fewer than requested {episodes}"
        )
    stop = int(starts[episodes]) if starts.size > episodes else len(records)

    squared_error = 0.0
    action_values = 0
    action_max_error = 0.0
    phase_counts: dict[int, int] = {}
    phase_max_error: dict[int, float] = {}
    reset_count = 0
    for record in records[:stop]:
        if float(record[36]) > 0.5:
            reset_fn()
            reset_count += 1
        actual = np.asarray(infer_fn(record[:32]), dtype=np.float32)
        expected = np.clip(record[32:36], -1.0, 1.0)
        if actual.shape != (4,):
            raise ValueError(f"callable returned shape {actual.shape}, expected (4,)")
        delta = actual - expected
        error = float(np.max(np.abs(delta)))
        action_max_error = max(action_max_error, error)
        squared_error += float(np.square(delta, dtype=np.float64).sum())
        action_values += delta.size
        gate = int(round(float(record[23]) * 6.0))
        phase_counts[gate] = phase_counts.get(gate, 0) + 1
        phase_max_error[gate] = max(phase_max_error.get(gate, 0.0), error)

    missing = [gate for gate in required_gates if phase_counts.get(gate, 0) == 0]
    rms = float(np.sqrt(squared_error / max(action_values, 1)))
    return {
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "episodes_replayed": reset_count,
        "records_replayed": int(stop),
        "phase_counts": {str(key): value for key, value in sorted(phase_counts.items())},
        "phase_max_error": {
            str(key): value for key, value in sorted(phase_max_error.items())
        },
        "missing_required_gates": missing,
        "action_rms_error": rms,
        "action_max_error": action_max_error,
        "action_atol": action_atol,
        "passed": not missing and action_max_error <= action_atol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--action-atol", type=float, default=5e-5)
    parser.add_argument("--json-path", type=Path)
    args = parser.parse_args()

    try:
        import policy_callable_six_gate_composite as composite
    except ModuleNotFoundError:
        from scripts import policy_callable_six_gate_composite as composite

    report = verify_dataset(
        args.dataset,
        infer_fn=composite.infer,
        reset_fn=composite.reset,
        episodes=args.episodes,
        action_atol=args.action_atol,
    )
    module_path = Path(composite.__file__).resolve()
    report["callable"] = str(module_path)
    report["callable_sha256"] = sha256_file(module_path)
    report["checkpoints"] = {}
    for label, name in (
        ("gate4", "PUFFER_POLICY_GATE4_CHECKPOINT_PATH"),
        ("gate5", "PUFFER_POLICY_GATE5_CHECKPOINT_PATH"),
        ("gate6", "PUFFER_POLICY_GATE6_CHECKPOINT_PATH"),
    ):
        path = Path(os.environ[name]).resolve()
        report["checkpoints"][label] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

